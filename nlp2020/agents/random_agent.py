from nlp2020.agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
   
    def __init__(self, 
                 action_dim):
        
        BaseAgent.__init__(self, 
                           action_dim, 
                           0, 
                           "RandomAgent")
        
    def act(self, state): return np.random.randint(self.action_dim)
    def update(self, i, state, action, next_state, reward): pass
    def reset(self): pass