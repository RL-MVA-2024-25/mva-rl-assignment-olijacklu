from gymnasium.wrappers import TimeLimit
from fast_env_py import FastHIVPatient
import numpy as np
import xgboost as xgb
import random
import os
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        
    def get_all(self):
        return list(zip(*self.data))
    
    def __len__(self):
        return len(self.data)

class ProjectAgent:
    def __init__(self):
        self.state_dim = 6
        self.action_dim = 4
        
        # Initialize XGBoost models for each action
        self.models = [None for _ in range(self.action_dim)]
        self.scalers = [StandardScaler() for _ in range(self.action_dim)]
        
        # Enhanced XGBoost parameters
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],  # Track multiple metrics
            'max_depth': 6,
            'eta': 0.05,  # Slower learning rate
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'min_child_weight': 3,
            'gamma': 0.1,  # Minimum loss reduction
            'lambda': 1.5,  # L2 regularization
            'alpha': 0.5,   # L1 regularization
            'tree_method': 'hist',
            'max_leaves': 64,  # Control tree complexity
            'seed': 42
        }
        
        # Enhanced training parameters
        self.exploration_steps = 30000  # More exploration
        self.num_boost_round = 200     # More trees
        self.gamma = 0.995            # Higher discount factor
        self.reward_scale = 1e-6
        
        # Early stopping parameters
        self.early_stopping_rounds = 10
        self.min_improvement = 0.001
        
        # Add epsilon-greedy exploration
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        self.epsilon = self.epsilon_start
        
        # Training parameters
        self.memory = ReplayBuffer(1000000)
        self.current_step = 0
        
    def act(self, observation, use_random=False):
        # Epsilon-greedy exploration
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
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, 
                         self.epsilon * self.epsilon_decay)
        return action
    
    def _prepare_fqi_dataset(self, states, actions, rewards, next_states, dones):
        """Prepare datasets for FQI training with reward scaling."""
        action_datasets = [[] for _ in range(self.action_dim)]
        action_targets = [[] for _ in range(self.action_dim)]
        
        # Scale down rewards to help with training stability
        scaled_rewards = rewards * self.reward_scale
        
        # Calculate max Q-values for next states
        next_q_values = np.zeros((len(states), self.action_dim))
        if self.models[0] is not None:  # If models exist
            for a in range(self.action_dim):
                next_states_scaled = self.scalers[a].transform(next_states)
                next_q_values[:, a] = self.models[a].predict(xgb.DMatrix(next_states_scaled))
        
        max_next_q = np.max(next_q_values, axis=1)
        
        # Prepare datasets for each action
        for i in range(len(states)):
            action = int(actions[i])
            target = scaled_rewards[i]  # Use scaled reward
            if not dones[i]:
                target += self.gamma * max_next_q[i]
            
            action_datasets[action].append(states[i])
            action_targets[action].append(target)
        
        return action_datasets, action_targets
    
    def _train_epoch(self):
        states, actions, rewards, next_states, dones = self.memory.get_all()
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        
        # Prepare datasets with bootstrapping
        n_samples = len(states)
        bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
        states = states[bootstrap_idx]
        actions = actions[bootstrap_idx]
        rewards = rewards[bootstrap_idx]
        next_states = next_states[bootstrap_idx]
        dones = dones[bootstrap_idx]
        
        action_datasets, action_targets = self._prepare_fqi_dataset(
            states, actions, rewards, next_states, dones
        )
        
        # Train models with early stopping and cross-validation
        for a in range(self.action_dim):
            if len(action_datasets[a]) > 0:
                X = np.array(action_datasets[a])
                y = np.array(action_targets[a])
                
                # Scale features
                X_scaled = self.scalers[a].fit_transform(X)
                
                # Create training and validation sets
                split_idx = int(0.8 * len(X))
                dtrain = xgb.DMatrix(X_scaled[:split_idx], label=y[:split_idx])
                dval = xgb.DMatrix(X_scaled[split_idx:], label=y[split_idx:])
                
                # Train with early stopping
                evals_result = {}
                self.models[a] = xgb.train(
                    self.xgb_params,
                    dtrain,
                    num_boost_round=self.num_boost_round,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    early_stopping_rounds=self.early_stopping_rounds,
                    evals_result=evals_result,
                    verbose_eval=False
                )
    
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
        # Track best models and their performance
        best_models = [None] * self.action_dim
        best_eval_reward = float('-inf')
        
        print("Starting enhanced initial exploration...")
        state, _ = env.reset()
        for step in tqdm(range(self.exploration_steps), desc="Initial exploration"):
            action = self.act(state, use_random=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            self.memory.append(state, action, reward, next_state, done)
            
            if done:
                state, _ = env.reset()
            else:
                state = next_state
            
            self.current_step += 1
            
            # Print progress every 1000 steps
            if (step + 1) % 1000 == 0:
                print(f"\nExploration step {step + 1}/{self.exploration_steps}")
                print(f"Buffer size: {len(self.memory)}")
        
        all_rewards = []
        eval_rewards = []
        running_avg = []
        
        print("\nStarting FQI training with enhanced monitoring...")
        for epoch in range(num_epochs):
            epoch_rewards = []
            
            # Collect episodes
            for episode in tqdm(range(episodes_per_epoch), 
                              desc=f"Epoch {epoch + 1}/{num_epochs}"):
                state, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action = self.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                    
                    self.memory.append(state, action, reward, next_state, done)
                    state = next_state
                
                epoch_rewards.append(episode_reward)
                all_rewards.append(episode_reward)
            
            # Train on all data
            self._train_epoch()
            
            # Evaluate
            eval_reward = self.evaluate(env, num_episodes=10)  # More evaluation episodes
            eval_rewards.append(eval_reward)
            
            # Update running average
            avg_reward = np.mean(epoch_rewards)
            running_avg.append(avg_reward)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Average Reward: {avg_reward:.2e}")
            print(f"Evaluation Reward: {eval_reward:.2e}")
            print(f"Epsilon: {self.epsilon:.3f}")
            
            # Save if best
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                best_models = [model.copy() if model else None for model in self.models]
                self.save(path="trained_models/best_model.pt")
        
        # Restore best models
        self.models = best_models
        return all_rewards, eval_rewards, running_avg
    
    def save(self, path):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save models and scalers
        save_dict = {
            'models': self.models,
            'scalers': self.scalers,
            'current_step': self.current_step
        }
        joblib.dump(save_dict, path)
    
    def load(self):
        path = "best_model.pt"
        save_dict = joblib.load(path)
        self.models = save_dict['models']
        self.scalers = save_dict['scalers']
        self.current_step = save_dict['current_step']
