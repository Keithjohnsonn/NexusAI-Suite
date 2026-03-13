import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Optional
from src.utils.logging import log

class QNetwork(nn.Module):
    """
    Core Deep Q-Network Architecture.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class DQNAgent:
    """
    A professional DQN agent implementation with Experience Replay.
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        lr: float = 1e-3, 
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Networks
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        log.info(f"DQN Agent initialized (states={state_dim}, actions={action_dim})")

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action based on epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(torch.argmax(q_values).item())

    def store_transition(self, s: np.ndarray, a: int, r: float, s_prime: np.ndarray, done: bool):
        """
        Stores experience in the replay buffer.
        """
        self.memory.append((s, a, r, s_prime, done))

    def update(self) -> Optional[float]:
        """
        Updates the network from a batch of sampled experiences.
        """
        if len(self.memory) < self.batch_size:
            return None
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q-values
        curr_q = self.q_net(states).gather(1, actions)
        
        # Max next Q-values from target net
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
            
        loss = nn.MSELoss()(curr_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def sync_target(self):
        """
        Synchronizes target network with behavior network.
        """
        self.target_net.load_state_dict(self.q_net.state_dict())
        log.debug("Target network synced.")
