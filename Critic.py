import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Critic(keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = Dense(200, activation='relu')
        self.fc2 = Dense(200, activation='relu')
        self.q = Dense(1, activation=None)

    @tf.function
    def __call__(self, input):
        action_value = self.fc1(input)
        action_value = self.fc2(action_value)

        q = self.q(action_value)
        return q

