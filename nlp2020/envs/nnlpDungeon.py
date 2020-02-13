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
        reward, done = BaseDungeon.step(self, action) 
        
        ob = self.dungeon_creator.visual_features
        return ob, reward, done, None
        
        
    def reset(self):
        BaseDungeon.reset(self) 
        return self.dungeon_creator.visual_features
        
        
    
    def close(self):
        print("close")
        
        
        
        