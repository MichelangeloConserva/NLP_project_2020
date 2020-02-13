import gym
import matplotlib.pyplot as plt

from gym import error, spaces, utils
from gym.utils import seeding

class BaseDungeon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 dungeon_creator, 
                 name):        
        self.dungeon_creator = dungeon_creator
        self.name = name
        
    
    def store_selection(self, selection, equipment):
        self.selection = selection
        self.equipment = equipment
        self.dungeon_creator.equipement_selected = selection
        
    def render(self):
        plt.imshow(self.dungeon_creator.visual_features)
        print(self.dungeon_creator.grid_world)
        
    def step(self, action):
        reward, done = self.dungeon_creator.receiving_action(action)
        
        return reward, done
        
        
    def reset(self):
        self.dungeon_creator.reset()
      
    def close(self):
        raise NotImplementedError("close")
        
        
    def __str__(self): return self.name    
    def __repr__(self): return self.name