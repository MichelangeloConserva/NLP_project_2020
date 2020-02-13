from nlp2020.agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
   
    def __init__(self, 
                 action_space_dim):
        
        BaseAgent.__init__(self, 
                           action_space_dim, 
                           0, 
                           "RandomAgent")
        
        
    def act(self, obs, reward, done):
        return np.random.randint(self.action_space_dim)
