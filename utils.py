import random

from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
from keras import backend as K

random.seed(123)


class Phi(object):
    """class pre-processes frames so can be inputted to deep q network"""
    def __init__(self, height, width, channels):
        self.frames = []
        self.height = height
        self.width = width
        self.channels = channels

        # add empty frames to start with
        for i in range(channels):
            zeros = np.zeros((height, width), dtype=np.uint8)
            self.frames.append(zeros)

    def add(self, screen):
        """Adds latest screen and returns latest four screens processed as nd array"""
        screen = screen.astype(np.uint8)  # save screen as 8 byte int
        processed_screen = resize(rgb2gray(screen), (self.height, self.width))

        self.frames.append(processed_screen)

        if len(self.frames) > self.channels:
            self.frames.pop(0)

        return np.array(self.frames)

    def latest_state(self):
        return np.array(self.frames)


def biased_flip(p):
    if random.random() < p:
        return True
    return False


def set_input(states):
    net_input = np.array(states, dtype=np.float32)
    return net_input / 255


def clipped_mean_squared_error(y_true, y_predict):
    clipped_squared_error = K.square(K.clip(y_true - y_predict, -1, 1))
    return K.mean(clipped_squared_error)