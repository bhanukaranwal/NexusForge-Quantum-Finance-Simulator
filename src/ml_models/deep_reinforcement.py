import asyncio
from typing import Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from pydantic import BaseModel, Field

from src.core.exceptions import MLModelException
from src.core.logging import logger


class RLConfig(BaseModel):
    state_dim: int = Field(default=20, description="State space dimension")
    action_dim: int = Field(default=3, description="Action space dimension")
    hidden_dim: int = Field(default=256, description="Hidden layer dimension")
    learning_rate: float = Field(default=3e-4, description="Learning rate")
    gamma: float = Field(default=0.99, description="Discount factor")
    epsilon_start: float = Field(default=1.0, description="Initial exploration rate")
    epsilon_end: float = Field(default=0.01, description="Final exploration rate")
    epsilon_decay: int = Field(default=1000, description="Exploration decay steps")
    batch_size: int = Field(default=64, description="Training batch size")
    memory_size: int = Field(default=100000, description="Experience replay buffer size")
    target_update_freq: int = Field(default=1000, description="Target network update frequency")
    algorithm: str = Field(default="DQN", description="RL algorithm: DQN, DDPG, PPO, A3C")


class PortfolioEnvironment(gym.Env):
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000, 
                 transaction_cost: float = 0.001, max_position: float = 1.0):
        super().__init__()
        
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0  # -1 to 1 (short to long)
        self.portfolio_value = initial_balance
        
        # Action space: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)
        
        # State space: market features + portfolio state
        n_features = len(data.columns) - 1  # Exclude target
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features + 3,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 50  # Start after warmup period
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        
        return self._get_observation(), {}
    
    def step(self, action):
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Get current and next prices
        current_price = self.data.iloc[self.current_step]['close']
        next_price = self.data.iloc[self.current_step + 1]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price, next_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}
    
    def _execute_action(self, action, current_price, next_price):
        old_portfolio_value = self.portfolio_value
        
        # Action mapping: 0=sell, 1=hold, 2=buy
        if action == 0:  # Sell
            target_position = max(self.position - 0.1, -self.max_position)
        elif action == 1:  # Hold
            target_position = self.position
        else:  # Buy
            target_position = min(self.position + 0.1, self.max_position)
        
        # Calculate position change
        position_change = target_position - self.position
        
        # Apply transaction costs
        transaction_cost = abs(position_change) * self.transaction_cost
        
        # Update position
        self.position = target_position
        
        # Calculate portfolio value change
        price_return = (next_price - current_price) / current_price
        position_return = self.position * price_return
        
        # Update portfolio value
        self.portfolio_value = self.portfolio_value * (1 + position_return) - transaction_cost * self.balance
        
        # Calculate reward (portfolio return)
        reward = (self.portfolio_value - old_portfolio_value) / old_portfolio_value
        
        return reward
    
    def _get_observation(self):
        if self.current_step >= len(self.data):
            # Return zeros if we're past the data
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Market features (excluding target)
        market_features = self.data.iloc[self.current_step].drop(['close']).values.astype(np.float32)
        
        # Portfolio state
        portfolio_state = np.array([
            self.position,
            self.portfolio_value / self.initial_balance,
            self.balance / self.initial_balance
        ], dtype=np.float32)
        
        return np.concatenate([market_features, portfolio_state])


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
    def forward(self, x):
        return self.network(x)


class DDPGActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Action values between -1 and 1
        )
        
    def forward(self, x):
        return self.network(x)


class DDPGCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
    
    def push(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return self.experience(*zip(*batch))
    
    def __len__(self):
        return len(self.buffer)


class DeepReinforcementLearning:
    def __init__(self, config: Union[RLConfig, Dict]):
        if isinstance(config, dict):
            config = RLConfig(**config)
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger.bind(component="DeepReinforcementLearning")
        
        # Initialize networks based on algorithm
        self._initialize_networks()
        
        # Experience replay buffer
        self.memory = ReplayBuffer(config.memory_size)
        
        # Training state
        self.steps_done = 0
        self.episode_rewards = []
        
    def _initialize_networks(self):
        if self.config.algorithm == "DQN":
            self.q_network = DQNNetwork(
                self.config.state_dim, self.config.action_dim, self.config.hidden_dim
            ).to(self.device)
            
            self.target_network = DQNNetwork(
                self.config.state_dim, self.config.action_dim, self.config.hidden_dim
            ).to(self.device)
            
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
            
        elif self.config.algorithm == "DDPG":
            self.actor = DDPGActor(
                self.config.state_dim, 1, self.config.hidden_dim  # 1D action for position
            ).to(self.device)
            
            self.critic = DDPGCritic(
                self.config.state_dim, 1, self.config.hidden_dim
            ).to(self.device)
            
            self.target_actor = DDPGActor(
                self.config.state_dim, 1, self.config.hidden_dim
            ).to(self.device)
            
            self.target_critic = DDPGCritic(
                self.config.state_dim, 1, self.config.hidden_dim
            ).to(self.device)
            
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())
            
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.learning_rate)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.learning_rate)

    async def train(
        self, 
        data: pd.DataFrame, 
        episodes: int = 1000,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, List[float]]:
        try:
            self.logger.info(f"Starting RL training with {self.config.algorithm}")
            
            # Prepare environment
            env = PortfolioEnvironment(data)
            
            episode_rewards = []
            episode_lengths = []
            losses = []
            
            for episode in range(episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
                episode_losses = []
                
                while True:
                    # Select action
                    action = await self._select_action(state, episode / episodes)
                    
                    # Execute action
                    next_state, reward, done, truncated, _ = env.step(action)
                    
                    # Store experience
                    self.memory.push(state, action, reward, next_state, done)
                    
                    state = next_state
                    episode_reward += reward
                    episode_length += 1
                    
                    # Train if enough experience
                    if len(self.memory) >= self.config.batch_size:
                        loss = await self._train_step()
                        if loss is not None:
                            episode_losses.append(loss)
                    
                    if done or truncated:
                        break
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                if episode_losses:
                    losses.append(np.mean(episode_losses))
                
                # Update target network
                if episode % self.config.target_update_freq == 0:
                    self._update_target_networks()
                
                if episode % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    self.logger.info(
                        f"Episode {episode}: Avg Reward: {avg_reward:.4f}, "
                        f"Episode Length: {episode_length}"
                    )
            
            self.episode_rewards = episode_rewards
            
            # Validation
            validation_results = {}
            if validation_data is not None:
                validation_results = await self._validate(validation_data)
            
            return {
                "episode_rewards": episode_rewards,
                "episode_lengths": episode_lengths,
                "losses": losses,
                "validation": validation_results,
            }
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise MLModelException(f"Training failed: {str(e)}")

    async def _select_action(self, state: np.ndarray, progress: float) -> int:
        if self.config.algorithm == "DQN":
            # Epsilon-greedy action selection
            epsilon = self.config.epsilon_end + (self.config.epsilon_start - self.config.epsilon_end) * np.exp(-1. * self.steps_done / self.config.epsilon_decay)
            self.steps_done += 1
            
            if random.random() > epsilon:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    return q_values.argmax().item()
            else:
                return random.randrange(self.config.action_dim)
        
        elif self.config.algorithm == "DDPG":
            # Add noise for exploration
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.actor(state_tensor)
                
                # Add Ornstein-Uhlenbeck noise
                noise_scale = 0.1 * (1 - progress)  # Decay noise over time
                noise = np.random.normal(0, noise_scale)
                action = action.cpu().numpy()[0] + noise
                
                # Convert continuous action to discrete (for environment compatibility)
                if action < -0.33:
                    return 0  # Sell
                elif action > 0.33:
                    return 2  # Buy
                else:
                    return 1  # Hold

    async def _train_step(self) -> Optional[float]:
        if len(self.memory) < self.config.batch_size:
            return None
        
        if self.config.algorithm == "DQN":
            return await self._train_dqn()
        elif self.config.algorithm == "DDPG":
            return await self._train_ddpg()
    
    async def _train_dqn(self) -> float:
        batch = self.memory.sample(self.config.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_state_batch).max(1)[0].detach()
        target_q_values = reward_batch + (self.config.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    async def _train_ddpg(self) -> float:
        batch = self.memory.sample(self.config.batch_size)
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.FloatTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Train Critic
        next_actions = self.target_actor(next_state_batch)
        target_q_values = self.target_critic(next_state_batch, next_actions).squeeze()
        target_q_values = reward_batch + (self.config.gamma * target_q_values * ~done_batch)
        
        current_q_values = self.critic(state_batch, action_batch.unsqueeze(1)).squeeze()
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Train Actor
        predicted_actions = self.actor(state_batch)
        actor_loss = -self.critic(state_batch, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return critic_loss.item() + actor_loss.item()
    
    def _update_target_networks(self):
        if self.config.algorithm == "DQN":
            self.target_network.load_state_dict(self.q_network.state_dict())
        elif self.config.algorithm == "DDPG":
            # Soft update
            tau = 0.001
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    async def _validate(self, validation_data: pd.DataFrame) -> Dict[str, float]:
        env = PortfolioEnvironment(validation_data)
        
        total_rewards = []
        total_returns = []
        
        for episode in range(10):  # Run 10 validation episodes
            state, _ = env.reset()
            episode_reward = 0
            initial_value = env.portfolio_value
            
            while True:
                action = await self._select_action(state, 1.0)  # No exploration
                next_state, reward, done, truncated, _ = env.step(action)
                
                state = next_state
                episode_reward += reward
                
                if done or truncated:
                    break
            
            total_rewards.append(episode_reward)
            total_returns.append((env.portfolio_value - initial_value) / initial_value)
        
        return {
            "avg_episode_reward": float(np.mean(total_rewards)),
            "std_episode_reward": float(np.std(total_rewards)),
            "avg_return": float(np.mean(total_returns)),
            "std_return": float(np.std(total_returns)),
            "sharpe_ratio": float(np.mean(total_returns) / (np.std(total_returns) + 1e-8)),
        }

    async def backtest(
        self, 
        data: pd.DataFrame,
        initial_balance: float = 100000
    ) -> Dict[str, Union[float, List[float]]]:
        try:
            env = PortfolioEnvironment(data, initial_balance)
            
            state, _ = env.reset()
            portfolio_values = [initial_balance]
            positions = []
            actions_taken = []
            
            while True:
                action = await self._select_action(state, 1.0)  # No exploration
                next_state, reward, done, truncated, _ = env.step(action)
                
                portfolio_values.append(env.portfolio_value)
                positions.append(env.position)
                actions_taken.append(action)
                
                state = next_state
                
                if done or truncated:
                    break
            
            # Calculate metrics
            portfolio_values = np.array(portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            total_return = (portfolio_values[-1] - initial_balance) / initial_balance
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            max_drawdown = np.max((np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values))
            
            return {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "final_balance": float(portfolio_values[-1]),
                "portfolio_values": portfolio_values.tolist(),
                "positions": positions,
                "actions": actions_taken,
            }
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {str(e)}")
            raise MLModelException(f"Backtesting failed: {str(e)}")

    def save_model(self, path: str) -> None:
        model_data = {
            "config": self.config.dict(),
            "steps_done": self.steps_done,
            "episode_rewards": self.episode_rewards,
        }
        
        if self.config.algorithm == "DQN":
            model_data["q_network"] = self.q_network.state_dict()
            model_data["target_network"] = self.target_network.state_dict()
        elif self.config.algorithm == "DDPG":
            model_data["actor"] = self.actor.state_dict()
            model_data["critic"] = self.critic.state_dict()
            model_data["target_actor"] = self.target_actor.state_dict()
            model_data["target_critic"] = self.target_critic.state_dict()
        
        torch.save(model_data, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        model_data = torch.load(path, map_location=self.device)
        
        self.config = RLConfig(**model_data["config"])
        self.steps_done = model_data.get("steps_done", 0)
        self.episode_rewards = model_data.get("episode_rewards", [])
        
        self._initialize_networks()
        
        if self.config.algorithm == "DQN":
            self.q_network.load_state_dict(model_data["q_network"])
            self.target_network.load_state_dict(model_data["target_network"])
        elif self.config.algorithm == "DDPG":
            self.actor.load_state_dict(model_data["actor"])
            self.critic.load_state_dict(model_data["critic"])
            self.target_actor.load_state_dict(model_data["target_actor"])
            self.target_critic.load_state_dict(model_data["target_critic"])
        
        self.logger.info(f"Model loaded from {path}")

    async def get_feature_importance(self, data: pd.DataFrame, n_samples: int = 1000) -> Dict[str, float]:
        """Calculate feature importance using perturbation analysis"""
        try:
            env = PortfolioEnvironment(data)
            base_rewards = []
            
            # Get baseline performance
            for _ in range(10):
                state, _ = env.reset()
                episode_reward = 0
                
                while True:
                    action = await self._select_action(state, 1.0)
                    next_state, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    state = next_state
                    
                    if done or truncated:
                        break
                
                base_rewards.append(episode_reward)
            
            baseline_reward = np.mean(base_rewards)
            
            # Test importance of each feature
            feature_importance = {}
            n_features = len(data.columns) - 1  # Exclude target
            
            for feature_idx in range(n_features):
                perturbed_rewards = []
                
                for _ in range(5):  # Fewer samples for speed
                    env = PortfolioEnvironment(data)
                    state, _ = env.reset()
                    episode_reward = 0
                    
                    while True:
                        # Perturb the specific feature
                        perturbed_state = state.copy()
                        perturbed_state[feature_idx] = np.random.normal(0, 1)  # Add noise
                        
                        action = await self._select_action(perturbed_state, 1.0)
                        next_state, reward, done, truncated, _ = env.step(action)
                        episode_reward += reward
                        state = next_state
                        
                        if done or truncated:
                            break
                    
                    perturbed_rewards.append(episode_reward)
                
                avg_perturbed_reward = np.mean(perturbed_rewards)
                importance = abs(baseline_reward - avg_perturbed_reward)
                feature_name = data.columns[feature_idx] if feature_idx < len(data.columns) - 1 else f"feature_{feature_idx}"
                feature_importance[feature_name] = importance
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {str(e)}")
            return {}
