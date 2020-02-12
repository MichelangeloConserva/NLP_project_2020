import gym
from gym import error, spaces, utils
from gym.utils import seeding

class BaseDungeon(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    raise NotImplementedError("__init__")
    
  def step(self, action):
    raise NotImplementedError("step")
    
  def reset(self):
    raise NotImplementedError("reset")
    
  def render(self, mode='human'):
    raise NotImplementedError("render")
    
  def close(self):
    raise NotImplementedError("close")