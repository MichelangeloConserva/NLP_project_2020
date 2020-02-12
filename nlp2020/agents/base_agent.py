


class BaseAgent:
    
    def __init__(self, 
                 action_space_dim, 
                 n_equip_can_take,
                 observation_space_dim, 
                 name):
        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim
        self.name = name

    def act(self, obs, reward, done):
        raise NotImplementedError("act")



    def __str__(self): return self.name    
    def __repr__(self): return self.name

