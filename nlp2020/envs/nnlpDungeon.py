import gym
from gym import error, spaces, utils
from gym.utils import seeding
from nlp2020.envs.base_dungeon import BaseDungeon


class nnlpDungeon(BaseDungeon):
    metadata = {'render.modes': ['human']}

    def __init__(self, dungeon_creator):
        BaseDungeon.__init__(self, dungeon_creator, 
                                   "NNLP-Dungeon")        
        
    
    def step(self, action):
        reward = self.dungeon_creator.result(action)
        return None, reward, False, None
        
        
        
    def reset(self):
        print("reset")
      
      
    def render(self, mode='human'):
        print("render")
        
        
    def close(self):
        print("close")
        
        
        
        