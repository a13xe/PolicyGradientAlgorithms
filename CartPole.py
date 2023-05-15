# !pip install keras-rl2
import os
import gym
import numpy as np
import tensorflow as tf
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory 
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import Adam


# модель для загрузки
model_to_load = "Models/cartpole_model_85000.h5"
# создание среды
env = gym.make("CartPole-v1") # render_mode="human" - параметр для визуализации обучения
# количество состояний
states = env.observation_space.shape[0] # print(states) # 4
# количество действий
actions = env.action_space.n # print(actions) # 2


if os.path.exists(model_to_load):
    # Загрузка созраненной в файл модели
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
else:
    # Создание новой модели нейронной машины
    model = Sequential()
    model.add(Flatten(input_shape = (1, states)))
    # 24 нейрона - первый слой (relu - rectified linear unit)
    model.add(Dense(24, activation="relu"))
    # 24 нейрона - второй слой
    model.add(Dense(24, activation="relu"))
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
    # компиляция модели с помощью оптимизатора Адама
    agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
    # шаги обучения
    agent.fit(env, nb_steps=85000, visualize=False, verbose=1)
    # Сохранение модели
    model.save("Models/cartpole_model_85000.h5")


# оценка модели - тестовые запуски
results = agent.test(env, nb_episodes=10, visualize=True)
# вывод среднего значения за эпизод
print(np.mean(results.history["episode_reward"]))


env.close()