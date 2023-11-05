import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.2):
        self.policy = Policy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def update(self, states, actions, rewards, log_probs, values):
        returns = self.compute_returns(rewards)
        advantages = self.compute_advantages(rewards, values)

        for _ in range(10):  # Number of optimization epochs
            new_log_probs = self.compute_log_probs(states, actions)
            ratio = torch.exp(new_log_probs - log_probs)

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surrogate1, surrogate2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def compute_advantages(self, rewards, values):
        returns = self.compute_returns(rewards)
        advantages = returns - values
        return advantages

    def compute_log_probs(self, states, actions):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        probs = self.policy(states)
        m = Categorical(probs)
        log_probs = m.log_prob(actions)
        return log_probs

# Example usage
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(state_dim, action_dim)

num_episodes = 1000
max_steps = 500

for episode in range(num_episodes):
    state = env.reset()
    rewards = []
    log_probs = []
    values = []

    for step in range(max_steps):
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        rewards.append(reward)
        log_probs.append(agent.compute_log_probs(state, action))
        values.append(agent.policy(torch.from_numpy(state).float().unsqueeze(0)))

        state = next_state

        if done:
            break

    agent.update(state, action, rewards, log_probs, values)

# Testing the trained agent
state = env.reset()
done = False

while not done:
    action = agent.select_action(state)
    state, reward, done, _ = env.step(action)
    env.render()

env.close()
