from random import sample, seed
import numpy as np

seed(123)


class ReplayMemory(object):

    def __init__(self, size):
        self.memories = []
        self.size = size

    def add(self, pre_state, action, reward, post_state, terminal):

        self.memories.append((pre_state, action, reward, post_state, terminal))

        if len(self.memories) > self.size:
            self.memories.pop(0)

    def getMemorySize(self):
        return len(self.memories)

    def getMinibatch(self, batch_size):
        return sample(self.memories, batch_size)
