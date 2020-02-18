import gym
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from itertools import permutations

from nlp2020.dung_descr_score import dungeon_description_generator




class DungeonCreator():
    def __init__(self):
        self.num_of_dungeon, self.n_equip = 5, 7
        self.n_equip_can_take = 2
        self.fully_informed = True
        
    def result(self, equipement_selected):
        
        if np.random.random() < self.score[equipement_selected].sum():
            return False
        return True
    
    def create_dungeon(self):
        self.dungeon_description, self.dung_type, self.score = dungeon_description_generator()
        
        if not self.fully_informed:
            self.dung_type = np.zeros(self.num_of_dungeon) 



class BaseDungeon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, name):        
        
        self.dungeon_creator = DungeonCreator()
        self.name = name
        
        # Create the permutations that represent the action selection
        n_equip = self.dungeon_creator.n_equip
        n_equip_can_take = self.dungeon_creator.n_equip_can_take
        assert n_equip < 10, "number of equipment too high, too slow"

        pp = n_equip_can_take * [1] + (n_equip-n_equip_can_take) * [0]
        self.action_to_selection = np.array(list(set(permutations(pp))))
        
        self.action_space = spaces.Discrete(len(self.action_to_selection))
        
    def is_fully_informed(self, b):
        self.dungeon_creator.fully_informed = b
        
    def step(self, action): 
        action = self.action_to_selection[action].astype(bool)
        done = self.dungeon_creator.result(action)
        reward = -self.n_mission_per_episode if done else +1
        done = done or self.cur_step == self.n_mission_per_episode
        self.cur_step += 1
        
        return reward, done
        
    def reset(self, n_mission_per_episode = 10):
        self.cur_step = 0        
        self.n_mission_per_episode = n_mission_per_episode
        self._max_episode_steps = n_mission_per_episode
        
        
    def next_dungeon(self):
        self.dungeon_creator.create_dungeon()
        
        
        
        
    def render(self, mode='human'): raise NotImplementedError("render")
    def close(self): raise NotImplementedError("close")
        
        
        
        
    def __str__(self): return self.name    
    def __repr__(self): return self.name
    
    
    
    
    
    