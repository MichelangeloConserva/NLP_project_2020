


class BaseAgent:
    
    def __init__(self, 
                 action_dim,
                 obs_dim, 
                 name):
        
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.name = name

        
    def update(self, i, state, action, next_state, reward): raise NotImplementedError("update")
    def reset(self): raise NotImplementedError("reset")
    def act(self, state): raise NotImplementedError("act")
    def start_episode(self, **args): pass
    def end_episode(self, **args): pass
    def before_act(self, **args): pass
    def __str__(self): return self.name    
    def __repr__(self): return self.name

