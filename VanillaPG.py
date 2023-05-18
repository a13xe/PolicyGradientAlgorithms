import gym
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from PPOconfig import PPOconfig
from PPOnetwork import Policy


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.fc = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x


env = gym.make('CartPole-v1')  # Replace 'CartPole-v1' with your desired environment
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

policy = Policy(input_size, output_size)


def train(policy, num_episodes, gamma, optimizer):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        trajectory = []
        
        while True:
            action_probs = policy(torch.Tensor(state))
            action = torch.multinomial(action_probs, 1).item()
            
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward))
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        returns = []
        cumulative_return = 0
        for _, _, reward in reversed(trajectory):
            cumulative_return = reward + gamma * cumulative_return
            returns.insert(0, cumulative_return)
        
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        optimizer.zero_grad()
        
        for (state, action, _), return_ in zip(trajectory, returns):
            action_probs = policy(torch.Tensor(state))
            loss = -torch.log(action_probs[action]) * return_
            loss.backward()
        
        optimizer.step()
        
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")


num_episodes = 500
gamma = 0.99
learning_rate = 0.01

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


train(policy, num_episodes, gamma, optimizer)
