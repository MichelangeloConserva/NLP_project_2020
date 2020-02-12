from nlp2020.agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    
    def __init__(self, 
                 action_space_dim, 
                 n_equip_can_take):
        
        BaseAgent.__init__(self, 
                           action_space_dim, 
                           n_equip_can_take,
                           0, 
                           "RandomAgent")
        
        
    def act(self, obs, reward, done, equipment_selection = False):
        
        if equipment_selection:
            a = np.random.rand(action_space_dim)
            return a >= sorted(a)[-2]
