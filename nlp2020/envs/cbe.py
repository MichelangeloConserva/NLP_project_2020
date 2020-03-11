import gym, random
from gym import error, spaces, utils
from gym.utils import seeding

from nlp2020.dung_descr_score import dungeon_description_generator



class cbe(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.name = "ContextualBandit"
        
        # TODO : link the max_length of sentence and the number of dungeon
        self.num_dung = spaces.Discrete(5)
        
        self.action_space = spaces.Discrete(7)
        
        self.reward_win = 1
        self.reward_die = -1
        
        
    def step(self, action, nlp = True):
        done = random.random() > self.probs_per_weapon[action]
        reward = self.reward_die if done else self.reward_win

        ob = self._sample(nlp) 
        return ob, reward, done, None
    
    def reset(self, nlp = True):
        return self._sample(nlp)


    def _sample(self, nlp):
        """
        Sample the new dungeon

        Returns
        -------
        None.

        """
        self.dungeon_description, self.dung_descr, self.probs_per_weapon = dungeon_description_generator()

        if nlp: return self.dungeon_description
        else:   return self.dung_descr 

    
    def render(self, mode='human'): pass
    def close(self): pass



































