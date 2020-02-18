from nlp2020.agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
   
    def __init__(self, action_dim):
        BaseAgent.__init__(self, action_dim, 0, "RandomAgent")
        
    def act(self, **args): return np.random.randint(self.action_dim)
    def save_model(self, **args):    pass
    def load_model(self, **args):    pass
    def reset(self, **args):         pass
    def update(self, **args):        pass