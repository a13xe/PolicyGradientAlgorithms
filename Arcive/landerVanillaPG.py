# This code defines the Lunar Lander environment, initializes the policy weights, 
# and implements the policy gradient algorithm for training. During training, 
# the agent chooses actions according to the current policy and collects gradients 
# of the log probabilities of the chosen actions. At the end of each episode, the 
# policy weights are updated using the collected gradients and the rewards obtained 
# during the episode. The process is repeated for a fixed number of episodes.
import gym
import numpy as np
     
env = gym.make('LunarLander-v2')

# hyperparameters
learning_rate = 0.01
discount_factor = 0.99
num_episodes = 500

# initialize policy weights
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
policy_weights = np.random.rand(state_dim, action_dim)

def choose_action(state, weights):
    # compute probabilities of each action
    logits = np.dot(state, weights)
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    # choose action randomly according to probabilities
    action = np.random.choice(action_dim, p=probabilities)
    return action, probabilities

def train():
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        gradients = []
        while True:
            action, probabilities = choose_action(state, policy_weights)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            # compute gradient of log probability
            gradient = np.outer(state, probabilities)
            gradient[action] -= state
            gradients.append(gradient)
            if done:
                # update policy weights
                for i in range(len(gradients)):
                    G = sum([discount_factor**t * r for t, r in enumerate(episode_rewards[i:])])
                    policy_weights += learning_rate * gradients[i] * G
                break
            state = next_state
    env.close()

train()
