import os
import torch
import numpy as np
from nltk.corpus import stopwords

try: stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    stopwords.words('english')


class BaseAgent:
    
    def __init__(self, action_dim, obs_dim, name, fully_informed, nlp):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.nlp = nlp
        self.fully_informed = fully_informed
        self.name = name + ("_" + \
            ("FullyInformed" if fully_informed else "NotInformed") + "_" +\
            ("NLP" if nlp else "NNLP") if name != "RandomAgent" else "")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def save_model(self, performance = None):   
        save_dir = "./logs_nlp2020/" 
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_dir + self.name + ".pth")   

        if performance is not None:
            np.savetxt(save_dir + self.name +".txt", performance)

    def load_model(self):
        checkpoint = torch.load("./logs_nlp2020/" + self.name + ".pth")
        self.reset()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def store_env_vars(self,**kwargs): self.__dict__.update(kwargs)
    def start_episode(self, **args):   pass
    def end_episode(self, **args):     pass
    def before_act(self, **args):      pass
    def act(self, **args):             raise  NotImplementedError("act")
    def reset(self, **args):           raise  NotImplementedError("reset")
    def update(self, **args):          raise  NotImplementedError("update")
    def __str__(self):                 return self.name    
    def __repr__(self):                return self.name
    def __hash__(self):                return hash(self.name)