import gym
import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size


def plot_training_results(rewards, algorithm, color):
    smoothed_rewards = moving_average(rewards, window_size=100)
    plt.plot(smoothed_rewards, color=color, label=algorithm)


def run_episode(env, policy):
    observation = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = policy(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        observation = next_observation

    return total_reward


def vpg(env, policy, learning_rate, num_episodes):
    rewards = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)  # Sample action from the policy function
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            state = next_state
        
        rewards.append(episode_reward)
    
    return rewards


def ppo(env, policy, learning_rate, num_episodes):
    rewards = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)  # Sample action from the policy function
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            state = next_state
        
        rewards.append(episode_reward)
    
    return rewards


def trpo(env, policy, learning_rate, num_episodes):
    rewards = []
    
    for episode in range(num_episodes):
        episode_reward = 0
        state = env.reset()
        done = False
        
        while not done:
            action = policy(state)  # Sample action from the policy function
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            
            state = next_state
        
        rewards.append(episode_reward)
    
    return rewards


# Policy function for random actions
class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def __call__(self, observation):
        return self.action_space.sample()


# Set up the environment
env = gym.make('CartPole-v1')  # render_mode="human"

# Training parameters
learning_rate = 0.01
num_episodes = 500

# Run VPG
random_policy = RandomPolicy(env.action_space)
vpg_rewards = vpg(env, random_policy, learning_rate, num_episodes)
plot_training_results(vpg_rewards, "VPG", color='blue')

# Run PPO
random_policy = RandomPolicy(env.action_space)
ppo_rewards = ppo(env, random_policy, learning_rate, num_episodes)
plot_training_results(ppo_rewards, "PPO", color='green')

# Run TRPO
random_policy = RandomPolicy(env.action_space)
trpo_rewards = trpo(env, random_policy, learning_rate, num_episodes)
plot_training_results(trpo_rewards, "TRPO", color='orange')

# Plot the combined training results
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Results - VPG, PPO, and TRPO')
plt.legend()

# Smooth the x-axis ticks for better readability
plt.xticks(np.arange(0, num_episodes, step=num_episodes//10))

plt.tight_layout()
plt.show()

env.close()
