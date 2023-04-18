import pygame
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense

import simulation
num_episodes = 10
max_timesteps = 10


# Initialize the Pygame module and the game environment
pygame.init()
game = simulation.LunarLander()

# Set up the policy network
policy_model = keras.Sequential(
    [
        Dense(32, activation="relu", input_shape=(game.observation_space,)),
        Dense(32, activation="relu"),
        Dense(game.action_space, activation="softmax"),
    ]
)

# Set up the optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# Set up the training loop
for episode in range(num_episodes):
    # Initialize the game environment
    observation = game.reset()
    episode_reward = 0

    with tf.GradientTape() as tape:
        for _ in range(max_timesteps):
            # Get the policy network's action predictions
            logits = policy_model(observation.reshape(1, -1))
            action = np.random.choice(game.action_space, p=np.squeeze(logits))

            # Take the action in the environment and record the results
            next_observation, reward, done, _ = game.step(action)
            episode_reward += reward

            # Calculate the loss
            log_prob = tf.math.log(tf.nn.softmax(logits))
            loss = -log_prob[:, action] * reward

            # Record the gradients
            grads = tape.gradient(loss, policy_model.trainable_variables)

            # Update the policy network
            optimizer.apply_gradients(zip(grads, policy_model.trainable_variables))

            # Update the observation for the next timestep
            observation = next_observation

            # Check if the episode is done and break out of the loop if it is
            if done:
                break

    # Print the results of the episode
    print(f"Episode {episode + 1}: Total reward = {episode_reward}")
