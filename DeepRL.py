##############################################################
#                                                            #
#    DRL - deep reinforcement learning                       #
#    Глубокое обучение с подкреплением для среды Cartpole    #
#                                                            #
##############################################################


# !pip install keras-rl2
import os
import gym
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory 
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam



model_to_load = "Models/DRLmodel_40800.h5" # модель для загрузки


# создание среды
env = gym.make("CartPole-v1") # render_mode="human" - параметр для визуализации обучения
states = env.observation_space.shape[0] # количество состояний # для среды Cartpole 4
actions = env.action_space.n # количество действий # для среды Cartpole 2


if os.path.exists(model_to_load): # Загрузка созраненной в файл модели
    model = load_model(model_to_load)
    agent = DQNAgent(
        model=model,
        memory=SequentialMemory(limit=50000, window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=0.01
    )
    agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
else: # Создание новой модели нейронной машины
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    model.add(Dense(24, activation="relu")) # 24 нейрона - первый слой (relu - rectified linear unit)
    model.add(Dense(24, activation="relu")) # 24 нейрона - второй слой
    model.add(Dense(actions, activation="linear"))
    # агент
    agent = DQNAgent(
        model=model,
        memory=SequentialMemory(limit=50000, window_length=1),
        policy=BoltzmannQPolicy(),
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=0.01    
    )
    # непосредственно обучение
    agent.compile(Adam(learning_rate=0.001), metrics=["mae"]) # компиляция модели с помощью оптимизатора Адама
    agent.fit(env, nb_steps=40800, visualize=False, verbose=1) # шаги обучения
    rewards = agent.fit(env, nb_steps=40800, visualize=False, verbose=1) # вознаграждения
    model.save("Models/DRLmodel_40800.h5") # Сохранение модели
    
    # Визуализация вознаграждений
    plt.plot(np.arange(1, len(rewards.history["episode_reward"]) + 1), rewards.history["episode_reward"])
    plt.style.use('seaborn-darkgrid')
    plt.title("DeepRL", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("reward", fontsize=12)
    plt.savefig('Rewards/DRL_rewards.png')



results = agent.test(env, nb_episodes=10, visualize=True) # оценка модели - тестовые запуски
print(np.mean(results.history["episode_reward"])) # вывод среднего значения за эпизод


env.close()