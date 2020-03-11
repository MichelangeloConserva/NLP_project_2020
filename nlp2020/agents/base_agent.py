import os
import torch
import numpy as np
from transformers import BertTokenizer
from nltk.corpus import stopwords
import pkgutil, re

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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # if nlp:

        #     # data = pkgutil.get_data(__package__, 'dictionary.txt')
        #     with open("dictionary.txt", "r") as f:
        #         self.vocabulary = f.read().splitlines()
            
        #     # Removing stopwords
        #     self.vocabulary = [word for word in self.vocabulary 
        #                        if word not in stopwords.words('english')]
            
        #     self.word2idx = {w: idx for (idx, w) in enumerate(self.vocabulary)}
        #     self.idx2word = {idx: w for (idx, w) in enumerate(self.vocabulary)}
            
        #     self.voc_size = len(self.vocabulary)


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


    def tokenize(self, sentence):
        if sentence is None: return None
        
        assert type(sentence) == str, sentence
        
        return  self.TEXT.process([self.TEXT.tokenize(sentence)], device = "cpu").squeeze().numpy()
        
    def filter_state(self, state, next_state):
        """
        Filter the state that is provided by the environment according to the 
        parameters of the agent.

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.
        next_state : TYPE
            DESCRIPTION.

        Returns
        -------
        state : TYPE
            DESCRIPTION.
        next_state : TYPE
            DESCRIPTION.

        """
        if not self.nlp:  
            if self.fully_informed:
                state = np.array(state, dtype = np.float)
                if not next_state is None: next_state = np.array(next_state, dtype = np.float)
            else:
                state = np.zeros(state.shape, dtype = np.float)
                if not next_state is None: next_state = np.zeros(next_state.shape, dtype = np.float)
        else:             
            state = self.tokenize(state)     
            if not next_state is None: next_state = self.tokenize(next_state)
        return state, next_state


    def start_episode(self, **args): pass
    def end_episode(self, **args):   pass
    def before_act(self, **args):    pass
    def act(self, **args):           raise  NotImplementedError("act")
    def reset(self, **args):         raise  NotImplementedError("reset")
    def update(self, **args):        raise  NotImplementedError("update")
    def __str__(self):               return self.name    
    def __repr__(self):              return self.name
    def __hash__(self):              return hash(self.name)
    # def __eq__(self, other):         return self.name == other.name
    # def __ne__(self, other):         return not(self == other)  
         
    
    
    
    
    