import os
import torch

class BaseAgent:
    
    def __init__(self, action_dim, obs_dim, name, fully_informed, nlp):
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.nlp = nlp
        self.fully_informed = fully_informed
        self.name = name + ("_" + \
            ("FullyInformed" if fully_informed else "NotInformed") + "_" +\
            ("NLP" if nlp else "NNLP") if name != "RandomAgent" else "")

    def save_model(self, save_dir, model):    
        
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        
        name = "model"
        if    self.nnlp: name += "_nnlp" 
        else:            name += "_nlp"     
    
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, save_dir + name)   


    def load_model(self, load_file, model_name):
        checkpoint = torch.load(load_file + ("_nnlp" if self.nnlp else ""))
        
        self.reset()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        model = getattr(self, )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

    def tokenize(self, sentence):
        token = [self.tokenizer.encode(sentence, add_special_tokens = True)]
        token = pad_sequences(token, maxlen=self.max_sentence_length, 
                              dtype="long", value=0, truncating="post", padding="post")
        
        return torch.tensor(token, device = self.device).long()




    def start_episode(self, **args): pass
    def end_episode(self, **args):   pass
    def before_act(self, **args):    pass
    def act(self, **args):           raise NotImplementedError("act")
    def reset(self, **args):         raise NotImplementedError("reset")
    def update(self, **args):        raise NotImplementedError("update")
    

    
    
    def __str__(self): return self.name    
    def __repr__(self): return self.name