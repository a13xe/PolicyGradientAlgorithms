# !pip install keras-rl2
import tensorflow as tf
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adama
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory 


# создание среды
env = gym.make("CartPole-v1", render_mode="human")

 
# количество состояний
states = env.observation_space.shape[0]
# print(states) # 4

# количество действий
actions = env.action_space.n
# print(actions) # 2


# модель нейронной машины
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

# компиляция модели с помощью оптимизатора Адама
agent.compile(Adam(learning_rate=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=True, verbose=1)

# оценка модели
results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))


env.close()