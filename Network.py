from osim.env import L2M2019Env
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

env = L2M2019Env(difficulty=3, visualize=False)
env.change_model(model='3D', difficulty=3)
observation = env.reset(project=True, obs_as_dict=True)
v_target = observation['v_tgt_field']
v_target = np.asarray(v_target).reshape(1,-1)

class Actor(object):
    def __init__(self, vtgt_field, hid_1=100, hid_2=100, hid_3=100):

        self.vtgt_field = vtgt_field
        self.model = Sequential([
            Dense(hid_1, input_shape=(np.shape(self.vtgt_field)), batch_size=None, activation='relu'),
            Dense(hid_2, activation='relu'),
            Dense(hid_3, activation='relu'),
            Dense(22, activation='relu')
        ])

    def get_model(self):
        return self.model


class Critic(object):
    def __init__(self, vtgt_field, hid_1=100, hid_2=100, hid_3=100):
        self.vtgt_field = vtgt_field
        self.model = Sequential([
            Dense(hid_1, input_shape=(np.shape(self.vtgt_field)), batch_size=None, activation='relu'),
            Dense(hid_2, activation='relu'),
            Dense(hid_3, activation='relu'),
            Dense(1, activation='none')
        ])

    def get_model(self):
        return self.model


netclass = Actor(v_target)
net1 = netclass.get_model()
net1.summary()