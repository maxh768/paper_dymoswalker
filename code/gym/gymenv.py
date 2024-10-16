from copy import deepcopy
import numpy as np
import gymnasium as gym
from gym.spaces import Discrete, Dict, Box

class Pendulum:
    def __init__(self, config=None):
        self.env = gym.make("InvertedPendulum-v5", render_mode="human")
        self.action_space = Discrete(1)
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info= self.env.step(action)
        return observation, reward, terminated, truncated, info

    def set_state(self, state):
        self.env.unwrapped.state = deepcopy(state)
        #obs = np.array(list(self.env.unwrapped.state))
        #return obs

    def get_state(self):
        #return deepcopy(self.env)
        return self.env.unwrapped._get_obs()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

