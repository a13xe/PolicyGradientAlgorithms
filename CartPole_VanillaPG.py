import gym
import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential


# Инициализация среды
env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n


# Определение сети политики
policy_network = Sequential([
    Dense(24, input_shape=(states,), activation='relu'),
    Dense(24, activation='relu'),
    Dense(actions, activation='softmax')
])


# Определение оптимизатора
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


# Определение цикла обучения
def train(policy_network, optimizer, num_episodes=1000, gamma=0.99):

    # Для каждого эпизода
    for i in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_states = []
        episode_actions = []
        episode_rewards = []

        # Запуск эпизода
        while True:
            # Выбор действий в соответствии с политикой
            action_probs = policy_network(tf.expand_dims(state, axis=0))
            action = np.random.choice(np.arange(actions), p=action_probs.numpy()[0])
            next_state, reward, done, _ = env.step(action)

            # Хранение состояния, действие и вознаграждения
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_reward += reward
            state = next_state

            # Проверка, закончился ли эпизод
            if done:
                # Вычисление дисконтированного вознаграждения
                discounted_rewards = np.zeros_like(episode_rewards)
                running_total = 0
                for t in reversed(range(len(episode_rewards))):
                    running_total = gamma * running_total + episode_rewards[t]
                    discounted_rewards[t] = running_total

                # Нормализация вознаграждений
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)

                # Вычисление градиентов
                with tf.GradientTape() as tape:
                    # Вычисление логарифмических вероятностей
                    logits = policy_network(np.array(episode_states))
                    log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=episode_actions, logits=logits)
                    # Вычисление потерь
                    loss = tf.reduce_mean(log_probs * discounted_rewards)

                # Вычисление градиентов и обновление политики
                grads = tape.gradient(loss, policy_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
                break

        # Вывести результата эпизода
        print(f'Episode {i}: Reward = {episode_reward}')

    env.close()


# Обучить нейронную сеть политики
train(policy_network, optimizer)
