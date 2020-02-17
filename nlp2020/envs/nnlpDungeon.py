import gym
from gym import error, spaces, utils
from gym.utils import seeding
from nlp2020.envs.base_dungeon import BaseDungeon


class nnlpDungeon(BaseDungeon):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        BaseDungeon.__init__(self, "NNLP-Dungeon")        
        
        self.observation_space = spaces.Discrete(self.dungeon_creator.num_of_dungeon)
                
        
    def step(self, action):
        action = self.action_to_selection[action].astype(bool)
        
        reward, done = self.dungeon_creator.result(action)
        ob = self.dungeon_creator.dung_type
        
        return ob, reward, done, None
        
        
    def reset(self):
        self.dungeon_creator.reset()
        return self.dungeon_creator.dung_type
      
      
    def render(self, mode='human'):
        pass
        
        
    def close(self):
        pass
        
        
        
        