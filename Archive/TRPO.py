import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the policy network
class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Function to estimate advantages using the Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, masks, gamma=0.99, tau=0.95):
    deltas = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)

    prev_value = 0
    prev_advantage = 0
    for t in reversed(range(len(rewards))):
        deltas[t] = rewards[t] + gamma * prev_value * masks[t] - values[t]
        advantages[t] = deltas[t] + gamma * tau * prev_advantage * masks[t]

        prev_value = values[t]
        prev_advantage = advantages[t]

    return advantages



# Function to collect trajectories using the current policy
def collect_trajectories(env, policy, timesteps):
    states = []
    actions = []
    rewards = []
    values = []
    masks = []

    state = env.reset()
    done = False
    t = 0

    while t < timesteps:
        state = torch.FloatTensor(state).to(device)
        action_logits = policy(state)
        action_dist = Categorical(logits=action_logits)
        action = action_dist.sample()

        next_state, reward, done, _ = env.step(action.cpu().numpy())

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        values.append(action_logits)
        masks.append(1 - done)

        state = next_state
        t += 1

        if done:
            state = env.reset()

    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = np.array(rewards, dtype=np.float32)
    values = torch.stack(values).detach().squeeze().cpu().numpy()
    masks = torch.FloatTensor(masks).to(device)

    return states, actions, rewards, values, masks

# Function to compute the surrogate loss
def surrogate_loss(states, actions, advantages, old_action_logits):
    action_logits = policy(states)
    action_dist = Categorical(logits=action_logits)
    entropy_loss = action_dist.entropy().mean()

    new_action_log_probs = action_dist.log_prob(actions)
    old_action_dist = Categorical(logits=old_action_logits)
    old_action_log_probs = old_action_dist.log_prob(actions)

    ratios = torch.exp(new_action_log_probs - old_action_log_probs)
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages

    policy_loss = -torch.min(surr1, surr2).mean()

    return policy_loss - 0.01 * entropy_loss

# TRPO training
def train(env, policy, max_iterations=500, timesteps_per_batch=4000, max_kl_divergence=0.01, damping_coefficient=0.1):
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    rewards_history = []

    for iteration in range(max_iterations):
        states, actions, rewards, values, masks = collect_trajectories(env, policy, timesteps_per_batch)
        advantages = compute_gae(rewards, values, masks)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        old_action_logits = policy(states).detach()

        for _ in range(10):
            loss = surrogate_loss(states, actions, advantages, old_action_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            kl_divergence = (old_action_logits - policy(states)).mean()

            if kl_divergence > max_kl_divergence:
                break

        episode_reward = sum(rewards)
        rewards_history.append(episode_reward)

        print(f"Iteration {iteration + 1} - Episode Reward: {episode_reward}")

    return rewards_history

# Function to visualize episodes
def visualize_episodes(env, policy, num_episodes=10):
    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            env.render()
            state = torch.FloatTensor(state).to(device)
            action_logits = policy(state)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample().cpu().numpy()
            next_state, _, done, _ = env.step(action)
            state = next_state

    env.close()

# Set up the environment
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Set up the policy network
policy = Policy(state_dim, action_dim).to(device)

# Train the policy
rewards_history = train(env, policy)

# Plot the training graph
plt.plot(rewards_history)
plt.xlabel('Iterations')
plt.ylabel('Episode Reward')
plt.title('TRPO CartPole Training')
plt.show()

# Visualize test episodes
visualize_episodes(env, policy)
