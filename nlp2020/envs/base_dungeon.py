import gym
import numpy as np

from gym import error, spaces, utils
from gym.utils import seeding
from itertools import permutations

from nlp2020.dung_descr_score import dungeon_description_generator


class DungeonCreator():
    def __init__(self):
        self.num_of_dungeon, self.n_equip = 5, 7
        self.fully_informed = True
        

        # old ##
        # self.weapons_next_dungeon = np.array([[0.60, 0.25, 0.05, 0.05, 0.05],
        #                                       [0.50, 0.10, 0.20, 0.10, 0.10],
        #                                       [0.40, 0.30, 0.10, 0.10, 0.10],
        #                                       [0.20, 0.25, 0.35, 0.10, 0.10],
        #                                       [0.20, 0.20, 0.20, 0.10, 0.30],
        #                                       [0.20, 0.20, 0.40, 0.10, 0.10],
        #                                       [0.10, 0.10, 0.10, 0.40, 0.30]])
        
        
        # for this we'll have
                ## if u choose weapon with 0 chance of success you got with very high prob to hard dungeons (so we end your agony quickly)
        self.weapons_next_dungeon = np.array([[0.60, 0.05, 0.05, 0.15, 0.10],
                                              [0.00, 0.50, 0.45, 0.05, 0.00],
                                              [0.40, 0.00, 0.10, 0.30, 0.20],
                                              [0.10, 0.35, 0.35, 0.10, 0.10],
                                              [0.35, 0.10, 0.05, 0.25, 0.25],
                                              [0.40, 0.10, 0.10, 0.20, 0.20],
                                              [0.10, 0.40, 0.30, 0.10, 0.10]])
        
    def result(self, equipement_selected):
        if np.random.random() < self.score[equipement_selected].sum():
            return False
        return True
    
    def create_dungeon(self, items):
        weight_vector = self.weapons_next_dungeon[items].mean(0)
        weight_vector = (weight_vector / weight_vector.sum())


        self.dungeon_description, self.dung_type, self.score = dungeon_description_generator(weight_vector)
        if not self.fully_informed: self.dung_type = np.zeros(self.num_of_dungeon) 

    def starting_dungeon(self):
        self.dungeon_description, self.dung_type, self.score = dungeon_description_generator()
        if not self.fully_informed: self.dung_type = np.zeros(self.num_of_dungeon) 



class BaseDungeon(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, name):        
        
        self.dungeon_creator = DungeonCreator()
        self.name = name
        
        
    def set_num_equip(self, n_equip_can_take = 2):
        
        # Create the permutations that represent the action selection
        n_equip = self.dungeon_creator.n_equip
        assert n_equip < 10, "number of equipment too high, too slow"

        pp = n_equip_can_take * [1] + (n_equip-n_equip_can_take) * [0]
        self.action_to_selection = np.array(list(set(permutations(pp))))
        
        self.action_space = spaces.Discrete(len(self.action_to_selection))        
        
        
    def is_fully_informed(self, b):
        self.dungeon_creator.fully_informed = b
        
    def step(self, action): 
        
        # Result of the action for the current dungeon
        action = self.action_to_selection[action].astype(bool)
        done = self.dungeon_creator.result(action)
        reward = -self.n_mission_per_episode if done else +1
        done = done or self.cur_step == self.n_mission_per_episode
        self.cur_step += 1
        
        # Sampling next dungeon
        self.dungeon_creator.create_dungeon(action)
        
        return reward, done
    
    
    def reset(self, n_mission_per_episode = 10):
        self.cur_step = 0        
        self.n_mission_per_episode = n_mission_per_episode
        self._max_episode_steps = n_mission_per_episode
        
        self.dungeon_creator.starting_dungeon()
        
        
    def render(self, mode='human'): raise NotImplementedError("render")
    def close(self): raise NotImplementedError("close")
        
        
        
        
    def __str__(self): return self.name    
    def __repr__(self): return self.name
    
    
    
