import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Set the random seed for reproducibility
np.random.seed(0)
tf.random.set_seed(0)

# Function to plot the training graph
def plot_training_graph(rewards, algorithm, name="graph.png"):
    plt.plot(rewards)
    plt.title(f"{algorithm} Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    # plt.show()
    plt.savefig(name)

# Vanilla Policy Gradient (VPG) Agent
class VPGAgent:
    def __init__(self, state_dim, action_dim, learning_rate, discount_factor):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_dim, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.model.predict(state).flatten()
        action = np.random.choice(self.action_dim, 1, p=action_probs)[0]
        return action

    def update_policy(self, states, actions, rewards):
        returns = self.calculate_returns(rewards)
        with tf.GradientTape() as tape:
            loss = self.calculate_loss(states, actions, returns)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def calculate_returns(self, rewards):
        returns = np.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + self.discount_factor * discounted_sum
            returns[t] = discounted_sum
        returns = (returns - np.mean(returns)) / np.std(returns)  # Normalize returns
        return returns

    def calculate_loss(self, states, actions, returns):
        action_probs = self.model(states, training=True)
        action_mask = tf.one_hot(actions, self.action_dim)
        selected_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
        loss = -tf.reduce_mean(tf.math.log(selected_action_probs) * returns)
        return loss

    def train(self, env, num_episodes):
        episode_rewards = []
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []

            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            episode_rewards.append(np.sum(rewards))
            returns = self.calculate_returns(rewards)

            self.update_policy(np.vstack(states), np.array(actions), returns)

            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_rewards[-1]}")

        return episode_rewards



# Trust Region Policy Optimization (TRPO) Agent
class TRPOAgent:
    def __init__(self, state_dim, action_dim, max_kl, damping):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_kl = max_kl
        self.damping = damping
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_dim, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.model.predict(state).flatten()
        action = np.random.choice(self.action_dim, 1, p=action_probs)[0]
        return action

    def surrogate_loss(self, states, actions, advantages):
        action_probs = self.model(states, training=True)
        action_mask = tf.one_hot(actions, self.action_dim)
        selected_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
        old_action_probs = tf.stop_gradient(selected_action_probs)
        ratio = selected_action_probs / old_action_probs
        surrogate_loss = -tf.reduce_mean(ratio * advantages)
        return surrogate_loss

    def hessian_vector_product(self, states, vector):
        with tf.GradientTape() as tape:
            action_probs = self.model(states, training=True)
            kl_divergence = tf.reduce_mean(tfp.distributions.kl_divergence(
                tfp.distributions.Categorical(probs=action_probs),
                tfp.distributions.Categorical(probs=action_probs)
            ))
            kl_gradients = tape.gradient(kl_divergence, self.model.trainable_variables)

        kl_hessian_vector_product = []
        for gradient, v in zip(kl_gradients, vector):
            if v is not None:
                product = tf.reduce_sum(gradient * tf.stop_gradient(v))
                with tf.GradientTape() as hessian_tape:
                    hessian_tape.watch(self.model.trainable_variables)
                    kl_hessian_vector_product.append(hessian_tape.gradient(product, self.model.trainable_variables))
            else:
                kl_hessian_vector_product.append(None)

        return kl_hessian_vector_product





    def conjugate_gradient(self, states, advantages, iterations=10):
        p = advantages
        r = self.hessian_vector_product(states, p)
        x = np.zeros_like(advantages, dtype=np.float32)
        for _ in range(iterations):
            alpha = np.dot(r, r) / np.dot(p, r)
            x += alpha * p
            r_new = r - alpha * self.hessian_vector_product(states, p)
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
        return x

    def update_policy(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            loss = self.surrogate_loss(states, actions, advantages)
        gradients = tape.gradient(loss, self.model.trainable_variables)

        # Compute Fisher vector product
        gradients_flat = tf.concat([tf.reshape(grad, [-1]) for grad in gradients], axis=0)
        gradient_vector_product = self.hessian_vector_product(states, gradients_flat)

        # Compute step direction using conjugate gradient
        step_direction = self.conjugate_gradient(states, gradient_vector_product)

        # Compute step size
        step_size = tf.sqrt(2 * self.max_kl / (tf.reduce_sum(step_direction * self.hessian_vector_product(states, step_direction)) + 1e-8))

        # Update policy
        new_variables = [old_var + step_size * step_dir for old_var, step_dir in zip(self.model.trainable_variables, step_direction)]
        self.model.set_weights(new_variables)

    def calculate_advantages(self, states, rewards, values):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + discounted_sum
            advantage = discounted_sum - values[t]
            returns[t] = discounted_sum
            advantages[t] = advantage
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Normalize advantages
        return returns, advantages

    def train(self, env, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            values = []

            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            # Calculate state values using a separate value function (e.g., a neural network)
            values = np.random.rand(len(states))

            returns, advantages = self.calculate_advantages(states, rewards, values)

            self.update_policy(states, actions, advantages)

            total_reward = np.sum(rewards)
            rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}")

        return rewards


# Proximal Policy Optimization (PPO) Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, learning_rate, clip_ratio, value_coef, entropy_coef):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.model = self.build_model()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, input_dim=self.state_dim, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        action_probs = self.model.predict(state).flatten()
        action = np.random.choice(self.action_dim, 1, p=action_probs)[0]
        return action

    def surrogate_loss(self, old_action_probs, states, actions, advantages):
        action_probs = self.model(states, training=True)
        action_mask = tf.one_hot(actions, self.action_dim)
        selected_action_probs = tf.reduce_sum(action_probs * action_mask, axis=1)
        old_action_probs = tf.stop_gradient(old_action_probs)
        ratio = selected_action_probs / old_action_probs
        surrogate_loss = -tf.reduce_mean(tf.minimum(
            ratio * advantages,
            tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        ))
        return surrogate_loss

    def value_loss(self, returns, values):
        returns = tf.reshape(returns, (-1,))
        values = tf.reshape(values, (-1,))
        return_loss = tf.reduce_mean(tf.square(returns - values))
        return return_loss

    def entropy_loss(self, action_probs):
        entropy_loss = -tf.reduce_mean(tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8), axis=1))
        return entropy_loss

    def update_policy(self, states, actions, old_action_probs, advantages, returns):
        with tf.GradientTape() as tape:
            surrogate_loss = self.surrogate_loss(old_action_probs, states, actions, advantages)
            values = self.model(states, training=True)
            value_loss = self.value_loss(returns, values)
            entropy_loss = self.entropy_loss(values)
            loss = surrogate_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def calculate_advantages(self, states, rewards, values):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + discounted_sum
            advantage = discounted_sum - values[t]
            returns[t] = discounted_sum
            advantages[t] = advantage
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)  # Normalize advantages
        return returns, advantages

    def train(self, env, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            values = []
            old_action_probs = []

            state = env.reset()
            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)

            # Calculate state values using a separate value function (e.g., a neural network)
            values = np.random.rand(len(states))

            returns, advantages = self.calculate_advantages(states, rewards, values)

            old_action_probs = self.model(states, training=True).numpy()

            self.update_policy(states, actions, old_action_probs, advantages, returns)

            total_reward = np.sum(rewards)
            rewards.append(total_reward)
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward}")

        return rewards


# Training VPG
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
learning_rate = 0.01
discount_factor = 0.99

vpg_agent = VPGAgent(state_dim, action_dim, learning_rate, discount_factor)
vpg_rewards = vpg_agent.train(env, num_episodes=500)

plot_training_graph(vpg_rewards, 'VPG', "graph_VPG.png")

# Training TRPO
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
max_kl = 0.01
damping = 0.1

trpo_agent = TRPOAgent(state_dim, action_dim, max_kl, damping)
trpo_rewards = trpo_agent.train(env, num_episodes=500)

plot_training_graph(trpo_rewards, 'TRPO', "graph_TRPO.png")

# Training PPO
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
learning_rate = 0.001
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0.01

ppo_agent = PPOAgent(state_dim, action_dim, learning_rate, clip_ratio, value_coef, entropy_coef)
ppo_rewards = ppo_agent.train(env, num_episodes=500)

plot_training_graph(ppo_rewards, 'PPO', "graph_PPO.png")
