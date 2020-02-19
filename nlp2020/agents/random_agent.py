from nlp2020.agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
   
    def __init__(self, action_dim):
        BaseAgent.__init__(self, action_dim, 0, "RandomAgent", False, False)
        
    def act(self, *args, **kwargs): return np.random.randint(self.action_dim)
    def save_model(self, *args, **kwargs):    pass
    def load_model(self, *args, **kwargs):    pass
    def reset(self, *args, **kwargs):         pass
    def update(self, *args, **kwargs):        pass