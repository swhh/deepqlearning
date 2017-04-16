import random
import numpy as np

from utils import biased_flip

random.seed(123)


class Agent(object):

    def __init__(self, replay_memory, dqnet, **kwargs):
        self.replay = replay_memory
        self.dqnet = dqnet
        self.num_actions = dqnet.num_actions
        self.init_explore_rate = kwargs.get('init_explore_rate', 1.0)
        self.final_explore_rate = kwargs.get('final_explore_rate', 0.1)
        self.final_explore_frame = kwargs.get('final_explore_frame', 1000000)
        self.train = kwargs.get('training_mode', True)
        self.replay_start_size = kwargs.get('replay_start_size', 50000)
        self.time_step = 0
        self.current_explore_rate = self.init_explore_rate
        self.decrement_explore_rate = (self.init_explore_rate - self.final_explore_rate) / (self.final_explore_frame - self.replay_start_size)
        self.current_loss = 0

    def get_action(self, state):

        if self.train and self.time_step < self.final_explore_frame and biased_flip(self.current_explore_rate):
            action = random.choice(range(self.num_actions))   # pick random action to explore state space

        else:
            scores = self.dqnet.predict(np.array([state]))
            action = np.argmax(scores[0])

        if self.train and self.replay_start_size <= self.time_step:
            mini_batch = self.replay.getMinibatch(self.dqnet.batch_size)
            self.current_loss = self.dqnet.train(mini_batch)

        if self.train and self.replay_start_size <= self.time_step < self.final_explore_frame:
            self.current_explore_rate -= self.decrement_explore_rate  # anneal explore rate

        self.time_step += 1
        return action

    def update_replay_memory(self, pre_state, action, reward, post_state, terminal):
        self.replay.add(pre_state, action, reward, post_state, terminal)

    def get_loss(self):
        return self.current_loss












