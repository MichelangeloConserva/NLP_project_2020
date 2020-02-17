


class BaseAgent:
    
    def __init__(self, 
                 action_space_dim,
                 observation_space_dim, 
                 name):
        
        self.action_space_dim = action_space_dim
        self.observation_space_dim = observation_space_dim
        self.name = name

    def act(self, state):
        raise NotImplementedError("act")

    def update(self, i, state, action, next_state, reward):
        raise NotImplementedError("update")


    def __str__(self): return self.name    
    def __repr__(self): return self.name

