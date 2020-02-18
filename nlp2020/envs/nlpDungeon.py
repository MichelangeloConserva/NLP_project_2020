import gym
from gym import error, spaces, utils
from gym.utils import seeding
from nlp2020.envs.base_dungeon import BaseDungeon








class nlpDungeon(BaseDungeon):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        BaseDungeon.__init__(self, "NNLP-Dungeon")        
        self.observation_space = spaces.Discrete(self.dungeon_creator.num_of_dungeon)
                
        
    def step(self, action):
        reward, done = BaseDungeon.step(self, action)
        
        # The observation is the description of the dungeon
        ob = self.dungeon_creator.dung_type
        
        return ob, reward, done, None
    
    
    def reset(self):
        BaseDungeon.reset(self)
        return self.dungeon_creator.dung_type
    
      
    def render(self, mode='human'): pass
    def close(self): pass
        
        
        
        
        