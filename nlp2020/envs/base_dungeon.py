import gym
from gym import error, spaces, utils
from gym.utils import seeding
from itertools import permutations

import numpy as np

class BaseDungeon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 dungeon_creator, 
                 name):        
        
        self.dungeon_creator = dungeon_creator
        self.name = name
        
        # Create the permutations that represent the action selection
        n_equip = dungeon_creator.n_equip
        n_equip_can_take = dungeon_creator.n_equip_can_take
        assert n_equip < 10, "number of equipment too high, too slow"

        pp = n_equip_can_take * [1] + (n_equip-n_equip_can_take) * [0]
        self.action_to_selection = np.array(list(set(permutations(pp))))
        
        self.action_space = spaces.Discrete(len(self.action_to_selection))
        
        
        

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