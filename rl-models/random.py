"""A random agent"""
import numpy as np


class RandomActor:
    def __init__(self, env, handle, *args, **kwargs):
        self.env = env
        self.handle = handle
        self.n_action = env.get_action_space(handle)[0]

    def infer_action(self, obs, *args, **kwargs):
        agents_num = obs[1].shape[0]
        actions = np.random.uniform(size=self.n_action*agents_num).astype('float32')
        return actions
