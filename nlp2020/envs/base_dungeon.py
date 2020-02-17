import gym
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from itertools import permutations
from nlp2020.dungeon_creator import DungeonCreator

    
equipment = ["sword", "bow", "water", "pickaxe"]
n_env = 2
action_space_dim = len(equipment)
n_equip_can_take = 2
effectivness_matrix = np.array(
    [[0.8, 0.1, 0.05, 0.05],
     [0.2, 0.1, 0.2, 0.5]])
effectivness_matrix = np.array(
    [[0.8, 0.2, 0.0, 0.0],
      [0.0, 0.9, 0.1, 0.0]])

n_equip_can_take = 2

class BaseDungeon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 name):        
        
        self.dungeon_creator = DungeonCreator(effectivness_matrix, 2)
        self.name = name
        
        # Create the permutations that represent the action selection
        n_equip = self.dungeon_creator.n_equip
        n_equip_can_take = self.dungeon_creator.n_equip_can_take
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