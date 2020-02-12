import gym
from gym import error, spaces, utils
from gym.utils import seeding

class BaseDungeon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 dungeon_creator, 
                 name):        
        
        self.dungeon_creator = dungeon_creator
        self.name = name
        
        
        
        
        
    def step(self, action):
      raise NotImplementedError("step")
    
    def reset(self):
        raise NotImplementedError("reset")
      
    def render(self, mode='human'):
        raise NotImplementedError("render")
      
    def close(self):
        raise NotImplementedError("close")
        
        
    def __str__(self): return self.name    
    def __repr__(self): return self.name