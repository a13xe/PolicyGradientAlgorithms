import torch
import torch.nn as nn


class Policy(nn.Module):
    def __init__(self, action_size, input_size=4):
        super(Policy, self).__init__()
        self.action_size = action_size  # Размер действий
        self.input_size = input_size  # Размер входных данных
        self.fc1 = nn.Linear(self.input_size, 24)  # Полносвязный слой с 24 выходами
        self.fc2 = nn.Linear(24, 24)  # Полносвязный слой с 24 выходами
        self.fc3_pi = nn.Linear(24, self.action_size)  # Полносвязный слой с выходами для каждого действия
        self.fc3_v = nn.Linear(24, 1)  # Полносвязный слой с 1 выходом для оценки ценности состояния
        self.tanh = nn.Tanh()  # Функция активации Tangent Hyperbolic
        self.relu = nn.ReLU()  # Функция активации ReLU
        self.softmax = nn.Softmax(dim=-1)  # Функция Softmax для получения вероятностного распределения

    def pi(self, x):
        x = self.relu(self.fc1(x))  # Применение ReLU к первому скрытому слою
        x = self.relu(self.fc2(x))  # Применение ReLU ко второму скрытому слою
        x = self.fc3_pi(x)  # Получение выходных значений для политики
        return self.softmax(x)  # Применение функции Softmax для получения вероятностного распределения действий

    def v(self, x):
        x = self.relu(self.fc1(x))  # Применение ReLU к первому скрытому слою
        x = self.relu(self.fc2(x))  # Применение ReLU ко второму скрытому слою
        x = self.fc3_v(x)  # Получение оценки ценности состояния
        return x