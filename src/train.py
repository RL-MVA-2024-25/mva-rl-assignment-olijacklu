from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import xgboost as xgb
import random
import os
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)

class ProjectAgent:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 4
        self.models = [None for _ in range(self.action_dim)]
        self.scalers = [StandardScaler() for _ in range(self.action_dim)]
        
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'max_depth': 6,
            'eta': 0.05,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,
            'lambda': 1.5,
            'alpha': 0.5,
            'tree_method': 'hist',
            'max_leaves': 64,
            'seed': 42
        }
        
        self.exploration_steps = 30000
        self.num_boost_round = 200
        self.gamma = 0.995
        self.reward_scale = 1e-6
        self.early_stopping_rounds = 10
        
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
    def act(self, observation, use_random=False):
        if use_random or random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            q_values = []
            obs_reshaped = observation.reshape(1, -1)
            for a in range(self.action_dim):
                if self.models[a] is not None:
                    obs_scaled = self.scalers[a].transform(obs_reshaped)
                    q_values.append(self.models[a].predict(xgb.DMatrix(obs_scaled))[0])
                else:
                    q_values.append(float('-inf'))
            action = np.argmax(q_values)
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return action

    def collect_transitions(self, steps, env, use_random=True):
        transitions = []
        state, _ = env.reset()
        
        for _ in range(steps):
            action = self.act(state, use_random)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            transitions.append((state, action, reward, next_state, done))
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
                
        return transitions

    def _prepare_fqi_dataset(self, transitions):
        states = np.vstack([t[0] for t in transitions])
        actions = np.array([t[1] for t in transitions])
        rewards = np.array([t[2] for t in transitions])
        next_states = np.vstack([t[3] for t in transitions])
        dones = np.array([t[4] for t in transitions])
        
        action_datasets = [[] for _ in range(self.action_dim)]
        action_targets = [[] for _ in range(self.action_dim)]
        
        scaled_rewards = rewards * self.reward_scale
        
        next_q_values = np.zeros((len(states), self.action_dim))
        if self.models[0] is not None:
            for a in range(self.action_dim):
                next_states_scaled = self.scalers[a].transform(next_states)
                next_q_values[:, a] = self.models[a].predict(xgb.DMatrix(next_states_scaled))
        
        max_next_q = np.max(next_q_values, axis=1)
        
        for i in range(len(states)):
            action = int(actions[i])
            target = scaled_rewards[i]
            if not dones[i]:
                target += self.gamma * max_next_q[i]
            
            action_datasets[action].append(states[i])
            action_targets[action].append(target)
        
        return action_datasets, action_targets
    
    def evaluate(self, env, num_episodes=5):
        total_reward = 0
        for _ in range(num_episodes):
            state, _ = env.reset()
            done = False
            while not done:
                action = self.act(state, use_random=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state
        return total_reward / num_episodes
    
    def train(self, env, num_epochs=6, episodes_per_epoch=200):
        best_models = [None] * self.action_dim
        best_eval_reward = float('-inf')
        
        print("Starting initial exploration...")
        all_transitions = self.collect_transitions(self.exploration_steps, env, use_random=True)
        
        all_rewards = []
        eval_rewards = []
        running_avg = []
        
        print("\nStarting FQI training...")
        for epoch in range(num_epochs):
            epoch_rewards = []
            epoch_transitions = []
            
            for episode in tqdm(range(episodes_per_epoch), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                episode_transitions = self.collect_transitions(200, env, use_random=False)
                episode_reward = sum(t[2] for t in episode_transitions)
                
                epoch_transitions.extend(episode_transitions)
                epoch_rewards.append(episode_reward)
                all_rewards.append(episode_reward)
            
            all_transitions.extend(epoch_transitions)
            
            n_samples = len(all_transitions)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_transitions = [all_transitions[i] for i in bootstrap_idx]
            
            action_datasets, action_targets = self._prepare_fqi_dataset(bootstrap_transitions)
            
            for a in range(self.action_dim):
                if len(action_datasets[a]) > 0:
                    X = np.array(action_datasets[a])
                    y = np.array(action_targets[a])
                    
                    X_scaled = self.scalers[a].fit_transform(X)
                    split_idx = int(0.8 * len(X))
                    dtrain = xgb.DMatrix(X_scaled[:split_idx], label=y[:split_idx])
                    dval = xgb.DMatrix(X_scaled[split_idx:], label=y[split_idx:])
                    
                    self.models[a] = xgb.train(
                        self.xgb_params,
                        dtrain,
                        num_boost_round=self.num_boost_round,
                        evals=[(dtrain, 'train'), (dval, 'val')],
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose_eval=False
                    )
            
            eval_reward = self.evaluate(env, num_episodes=10)
            eval_rewards.append(eval_reward)
            avg_reward = np.mean(epoch_rewards)
            running_avg.append(avg_reward)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average Reward: {avg_reward:.2e}")
            print(f"Evaluation Reward: {eval_reward:.2e}")
            print(f"Epsilon: {self.epsilon:.3f}")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_models = [model.copy() if model else None for model in self.models]
                self.save(path="trained_models/best_model.pt")
        
        self.models = best_models
        return all_rewards, eval_rewards, running_avg
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {
            'models': self.models,
            'scalers': self.scalers
        }
        joblib.dump(save_dict, path)
    
    def load(self):
        path = os.getcwd() + "/best_model.pt"
        save_dict = joblib.load(path)
        self.models = save_dict['models']
        self.scalers = save_dict['scalers']
