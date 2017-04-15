import random

from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np

from keras import backend as K

random.seed(123)


class Phi(object):
    """class pre-processes frames so can be inputted to deep q network"""
    def __init__(self, frame_num, new_height, new_width, channels, height, width):
        self.frames = []
        self.height = height
        self.width = width
        self.channels = channels
        self.frame_num = frame_num
        self.new_height = new_height
        self.new_width = new_width

        # add empty frames to start with
        for i in range(frame_num):
            zeros = np.zeros((height, width, channels), dtype=np.uint8)
            self.frames.append(zeros)

    def add(self, screen):
        """Adds latest screen and returns latest four processed screens processed as nd array"""

        self.frames.append(screen.astype(np.uint8))  # store original screen as 8 bit int
        self.frames.pop(0)

        processed_screens = [resize(rgb2gray(screen), (self.new_height, self.new_width)) for screen in self.frames]
        return np.array(processed_screens, dtype=np.float32)

    def reset(self):
        self.frames = []
        for i in range(self.frame_num):
            zeros = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
            self.frames.append(zeros)


def biased_flip(p):
    if random.random() < p:
        return True
    return False


def huber_loss(y_true, y_pred):
    x = y_true - y_pred
    return K.switch(K.abs(x) < 1.0, 0.5 * K.square(x), K.abs(x) - 0.5)

