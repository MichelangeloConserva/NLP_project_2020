import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.distributions import Categorical

from nlp2020.agents.base_agent import BaseAgent
from nlp2020.architectures import NLP_ActorCritic, ActorCritic
from nlp2020.utils import categorical_accuracy

# https://github.com/seungeunrho/minimalRL/blob/master/acer.py

class ACER_agent(BaseAgent):
    def __init__(self, obs_dim, action_dim, vocab_size, embedding_dim, 
                 n_filters, filter_sizes, dropout, pad_idx,TEXT,
                 fully_informed = True, nlp = True, learning_rate = 0.0002,
                 gamma = 0.98, c = 1.0, sl_separated_rl = True, only_rl = False,
                 dp_rl = 0):  
            
        BaseAgent.__init__(self, action_dim, obs_dim, "ACERAgent", fully_informed, nlp)            
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.c = c
        self.vocab_size = vocab_size
        self.embedding_dim  = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.pad_idx = pad_idx
        self.TEXT = TEXT
        self.sl_separated_rl = sl_separated_rl
        self.only_rl = only_rl
        self.dp_rl = dp_rl
        
        if nlp:
            if only_rl: self.name += "_only_RL"
            elif sl_separated_rl: self.name += "_SL_sep_RL"
            else: self.name += "_SL_both_RL"
        
        if dp_rl != 0: self.name += "_dropout"
        
        
        self.reset()
        self.criterion = nn.CrossEntropyLoss()
    
    def NLP_process(self, batch):
        self.optimizer_SL.zero_grad()
        predictions = self.model.NLP(batch.text)
        loss = self.criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        
        loss_SL = acc_SL = 0
        if not self.only_rl:
            loss.backward(retain_graph= not self.sl_separated_rl)#not self.sl_rl)
            self.optimizer_SL.step()
            loss_SL = loss.item()
            acc_SL = acc.item()
        
        batch.label = batch.label.cpu().numpy().tolist()
        
        if self.sl_separated_rl: state = predictions.detach() 
        else:                    state = predictions    
        
        return state, loss_SL, acc_SL
        
    def NNLP_process(self, batch):
        batch.label = batch.label.cpu().numpy()
        state = np.zeros((len(batch),self.obs_dim))
        if self.fully_informed: state[np.arange(len(batch)), batch.label] = 1
        return torch.from_numpy(state).float().to(self.device), 0, 0
    
    def act(self, state, labels, test):
        
        if self.nlp: model = self.model.RL
        else:        model = self.model        
        
        with torch.no_grad(): prob = model.pi(state)#.cpu().numpy()

        if test: actions = prob.cpu().numpy().argmax(1)
        else:    actions = Categorical(prob).sample().cpu().numpy()    
    
        dead = np.random.random(len(actions)) > self.weapon_in_dung_score[labels,actions]
        r = np.ones(len(dead)) * self.reward_win
        r[dead] = self.reward_die    
    
        if test: return r.tolist()
        return actions, r, prob, dead
    
    def act_and_train(self, batch, test = False, return_actions = False):
        
        if self.nlp: state, loss_SL, acc_SL = self.NLP_process(batch)
        else:        state, loss_SL, acc_SL = self.NNLP_process(batch)
    
        if test: return self.act(state, batch.label, test)
    
        actions, r, prob, dead = self.act(state, batch.label, test)
        
        if self.nlp: model = self.model.RL
        else:        model = self.model
        
        s = state
        a = torch.from_numpy(actions).to(self.device).view(-1,1)
        prob = prob.to(self.device)
        dead = torch.from_numpy(dead).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        
        q = model.q(s)
        q_a = q.gather(1,a)
        pi = model.pi(s, softmax_dim = 1)
        pi_a = pi.gather(1,a)
        v = (q * pi).sum(1).unsqueeze(1).detach()
        
        rho = pi.detach()/prob
        rho_a = rho.gather(1,a)
        rho_bar = rho_a.clamp(max=self.c)
        correction_coeff = (1-self.c/rho).clamp(min=0)
        
        q_ret = v[-1] * dead[-1]
        q_ret_lst = []
        for i in reversed(range(len(r))):
            q_ret = r[i] + self.gamma * q_ret
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]
            
            # if is_first[i] and i!=0:
            #     q_ret = v[i-1] * dead[i-1] # When a new sequence begins, q_ret is initialized  
                
        q_ret_lst.reverse()
        q_ret = torch.tensor(q_ret_lst, dtype=torch.float, device=self.device).unsqueeze(1)
        
        loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v) 
        loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach()-v) # bias correction term
        loss = (loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)).mean()
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=not self.sl_separated_rl)
        for name,param in self.model.named_parameters(): 
            if not param.grad is None: param.grad.data.clamp_(-1, 1)
        self.optimizer.step()    
    
        if return_actions: 
            return loss_SL, acc_SL, r.tolist(), actions
        else:
            return loss_SL, acc_SL, r.tolist()
        
        
    def reset(self):
        
        if not self.nlp:
            self.model = ActorCritic(self.obs_dim, self.action_dim, self.dp_rl).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)    
        else:
            k = self.obs_dim
            if self.only_rl: k = 512
            
            self.model = NLP_ActorCritic(k, self.action_dim,
                   self.vocab_size, self.embedding_dim, self.n_filters, self.filter_sizes, 
                   k, self.dropout, self.pad_idx, self.dp_rl).to(self.device)
            self.optimizer = optim.Adam(
                [
                    {'params': self.model.RL.parameters()},
                    {'params': self.model.NLP.parameters(), 'lr': self.learning_rate}
                ],    
            lr=self.learning_rate)
            
            self.model.NLP.embedding.weight.data[self.pad_idx] = torch.zeros(self.embedding_dim)
            self.optimizer_SL = optim.Adam(self.model.NLP.parameters(), lr=self.learning_rate)    
            
        self.model.to(self.device)