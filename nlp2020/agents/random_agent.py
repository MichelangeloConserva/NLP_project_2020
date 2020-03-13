from nlp2020.agents.base_agent import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    
    def __init__(self, action_dim):      
        BaseAgent.__init__(self, action_dim, 7, "Random", False, False)            
        self.action_dim = action_dim
    
    def act(self, state, labels, test = False):
        global weapon_in_dung_score, reward_win, reward_die
        
        actions = np.random.randint(0,self.action_dim, len(labels))   
        dead = np.random.random(len(actions)) > weapon_in_dung_score[labels,actions]
        r = np.ones(len(dead)) * reward_win
        r[dead] = reward_die    
    
        return actions, r, dead
    
    def act_and_train(self, batch, test = False):
        batch.label = batch.label.cpu().numpy()
        actions, r, dead = self.act(None, batch.label)
        if test: return r.tolist()
        return 0, 0, r.tolist()
        
    def reset(self):  pass