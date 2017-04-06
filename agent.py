import random
import numpy as np

from utils import biased_flip

np.random.seed(123)
random.seed(123)


class Agent(object):

    def __init__(self, replay_memory, dqnet, phi, **kwargs):
        self.replay = replay_memory
        self.dqnet = dqnet
        self.num_actions = dqnet.num_actions
        self.init_explore_rate = kwargs.get('init_explore_rate', 1.0)
        self.final_explore_rate = kwargs.get('final_explore_rate', 0.1)
        self.final_explore_frame = kwargs.get('final_explore_frame', 10000)
        self.train = kwargs.get('training_mode', True)
        self.time_step = 0
        self.current_explore_rate = self.init_explore_rate
        self.decrement_explore_rate = (self.init_explore_rate - self.final_explore_rate) / self.final_explore_frame
        self.phi = phi

    def get_action(self, observation):

        state = self.phi.add(observation)  # get right input for dq network

        if self.time_step < self.final_explore_frame and biased_flip(self.current_explore_rate):
            action = random.choice(range(self.num_actions))   # pick random action to explore state space
        else:
            scores = self.dqnet.predict(np.array([state]))
            action = np.argmax(scores[0])

        if self.train and self.replay.getMemorySize() > self.dqnet.batch_size:
            mini_batch = self.replay.getMinibatch(self.dqnet.batch_size)
            self.dqnet.train(mini_batch)

        if self.time_step < self.final_explore_frame:
            self.current_explore_rate -= self.decrement_explore_rate  # anneal explore rate

        self.time_step += 1
        return action

    def update_replay_memory(self, pre_state, action, reward, post_state, terminal):
        pre_state = self.phi.add(pre_state)
        post_state = self.phi.add(post_state)
        self.replay.add(pre_state, action, reward, post_state, terminal)












