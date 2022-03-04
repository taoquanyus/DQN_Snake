import random
from collections import deque

import numpy as np


class ReplayBuffer:

    """ 经验回放池 """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # 队列，先进先出

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done)) # 将数据加入buffer

    def sample(self, batch_size): # 从buffer中采样数据，数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self): # 目前buffer中数据的数量
        return len(self.buffer)