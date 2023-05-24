import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Set the random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PolicyGradientAgent:
    def __init__(self, env, learning_rate=0.01, discount_factor=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Create the policy network
        self.policy_model = self.build_policy_model()
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def build_policy_model(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
                tf.keras.layers.Dense(24, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax'),
            ]
        )
    
    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        probabilities = self.policy_model.predict(state).flatten()
        return np.random.choice(range(self.env.action_space.n), p=probabilities)
    
    def compute_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        cumulative_reward = 0
        for t in reversed(range(len(rewards))):
            cumulative_reward = rewards[t] + self.discount_factor * cumulative_reward
            discounted_rewards[t] = cumulative_reward
        return discounted_rewards
    
    def train(self, num_episodes=500, batch_size=32, algorithm='vpg'):
        all_rewards = []
        average_rewards = []
        
        for episode in range(num_episodes):
            episode_rewards = []
            episode_states = []
            episode_actions = []
            
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                
                state = next_state
            
            all_rewards.append(sum(episode_rewards))
            average_rewards.append(np.mean(all_rewards[-10:]))
            
            states = np.array(episode_states)
            actions = np.array(episode_actions)
            discounted_rewards = self.compute_discounted_rewards(episode_rewards)
            
            if algorithm == 'vpg':
                self.vanilla_policy_gradient_update(states, actions, discounted_rewards)
            elif algorithm == 'ppo':
                self.proximal_policy_optimization_update(states, actions, discounted_rewards)
            elif algorithm == 'trpo':
                self.trust_region_policy_optimization_update(states, actions, discounted_rewards)
            
            if episode % 500 == 0:
                print(f"Episode {episode}/{num_episodes}, Reward: {all_rewards[-1]}, Avg Reward: {average_rewards[-1]}")
        
        return all_rewards, average_rewards
    
    def vanilla_policy_gradient_update(self, states, actions, discounted_rewards):
        with tf.GradientTape() as tape:
            logits = self.policy_model(states, training=True)
            action_masks = tf.one_hot(actions, self.env.action_space.n)
            log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis=1)
            loss = -tf.reduce_mean(log_probs * discounted_rewards)
        
        gradients = tape.gradient(loss, self.policy_model.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))
    
    def proximal_policy_optimization_update(self, states, actions, discounted_rewards, epsilon=0.2, epochs=10):
        old_policy = tf.keras.models.clone_model(self.policy_model)
        old_policy.set_weights(self.policy_model.get_weights())
        
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                logits = self.policy_model(states, training=True)
                action_masks = tf.one_hot(actions, self.env.action_space.n)
                log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis=1)
                
                old_logits = old_policy(states, training=True)
                old_log_probs = tf.reduce_sum(action_masks * tf.math.log(old_logits), axis=1)
                
                ratio = tf.exp(log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
                surrogate_loss = -tf.reduce_mean(tf.minimum(ratio * discounted_rewards, clipped_ratio * discounted_rewards))
            
            gradients = tape.gradient(surrogate_loss, self.policy_model.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))
            
    def trust_region_policy_optimization_update(self, states, actions, discounted_rewards, delta=0.01, epochs=10):
        for _ in range(epochs):
            with tf.GradientTape() as tape:
                logits = self.policy_model(states, training=True)
                action_masks = tf.one_hot(actions, self.env.action_space.n)
                log_probs = tf.reduce_sum(action_masks * tf.math.log(logits), axis=1)
                
                old_logits = self.policy_model(states, training=False)
                old_log_probs = tf.reduce_sum(action_masks * tf.math.log(old_logits), axis=1)
                
                ratio = tf.exp(log_probs - old_log_probs)
                surrogate_loss = -tf.reduce_mean(ratio * discounted_rewards)
                
                kl_divergence = tf.reduce_mean(tf.reduce_sum(old_logits * (tf.math.log(old_logits + 1e-10) - tf.math.log(logits + 1e-10)), axis=1))
                kl_penalty = tf.reduce_mean(tf.square(kl_divergence - delta))
                loss = surrogate_loss + kl_penalty
            
            gradients = tape.gradient(loss, self.policy_model.trainable_variables)
            self.policy_optimizer.apply_gradients(zip(gradients, self.policy_model.trainable_variables))

# Initialize the environment
env = gym.make('CartPole-v1')

# Initialize the agent
agent = PolicyGradientAgent(env)

# Train the agent using Vanilla Policy Gradient (VPG)
vpg_rewards, vpg_avg_rewards = agent.train(num_episodes=500, algorithm='vpg')

# Train the agent using Proximal Policy Optimization (PPO)
ppo_rewards, ppo_avg_rewards = agent.train(num_episodes=500, algorithm='ppo')

# Train the agent using Trust Region Policy Optimization (TRPO)
trpo_rewards, trpo_avg_rewards = agent.train(num_episodes=500, algorithm='trpo')

# Plot the training graphs
plt.figure(figsize=(12, 6))
plt.plot(vpg_rewards, label='VPG Rewards')
# plt.plot(vpg_avg_rewards, label='VPG Average Rewards')
plt.plot(ppo_rewards, label='PPO Rewards')
# plt.plot(ppo_avg_rewards, label='PPO Average Rewards')
plt.plot(trpo_rewards, label='TRPO Rewards')
# plt.plot(trpo_avg_rewards, label='TRPO Average Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Performance')
plt.legend()
plt.show()
plt.savefig('qwer.png')
