from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras import backend as K

from utils import set_input, clipped_mean_squared_error
import time

np.random.seed(123)


class DeepQNetwork(object):

    def __init__(self, num_actions, input_shape, **kwargs):
        # set params
        self.time_step = 0
        self.num_actions = num_actions
        self.discount = kwargs.get('discount', 0.99)
        self.c = kwargs.get('c', 10000)  # how many time steps between target network updates
        self.batch_size = kwargs.get('batch_size', 32)
        self.hist_length = kwargs.get('hist_length', 4)

        # build conv model
        q_model = Sequential()
        q_model.add(Convolution2D(32, (8, 8), activation='relu', input_shape=input_shape,
                                  dim_ordering='th', strides=(4, 4)))
        q_model.add(BatchNormalization())
        q_model.add(Convolution2D(64, (4, 4), activation='relu', strides=(2, 2)))
        q_model.add(BatchNormalization())
        q_model.add(Convolution2D(64, (3, 3), activation='relu', strides=(1, 1)))
        q_model.add(BatchNormalization())
        q_model.add(Flatten())
        q_model.add(Dense(512, activation='relu'))
        q_model.add(BatchNormalization())
        q_model.add(Dense(num_actions, activation='linear'))
        q_model.compile(optimizer='RMSprop', loss=clipped_mean_squared_error)

        self.q_model = q_model
        self.q_target_model = Sequential.from_config(q_model.get_config())  # create target model

    def train(self, minibatch):
        states, rewards, actions, future_states, terminals = zip(*minibatch)
        rewards, actions, terminals = np.array(rewards), np.array(actions, dtype=np.uint8), np.array(terminals)
        rewards = np.clip(rewards, -1, 1)   # clip rewards

        q_values = self.predict(states) # predicted q values for all actions
        target_q_values = self.predict(future_states, is_q_model=False)
        q_values[np.arange(len(minibatch)), actions] = rewards
        q_values[np.arange(len(minibatch)), actions] += self.discount * ~terminals * np.max(target_q_values, axis=1)
        # if action didn't result in terminal state, add q max action

        if self.time_step % 10000 == 0:
            verbose = 1
        else:
            verbose = 0

        self.q_model.fit(set_input(states), q_values, batch_size=self.batch_size, nb_epoch=1, verbose=verbose)
        self.time_step += 1

        if self.time_step % self.c == 0:
            self.q_target_model.set_weights(self.q_model.get_weights())

    def predict(self, states, is_q_model=True):
        if is_q_model:
            return self.q_model.predict(set_input(states))

        return self.q_target_model.predict(set_input(states))

    def load_model(self, filename):
        self.q_model.load_weights(filename)

    def save_model(self, filename):
        self.q_model.save_weights(filename)