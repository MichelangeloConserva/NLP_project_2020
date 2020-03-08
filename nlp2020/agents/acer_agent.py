import gym,os
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from nlp2020.agents.base_agent import BaseAgent
from nlp2020.architectures import NLP_ActorCritic, ReplayBuffer, ActorCritic

# https://github.com/seungeunrho/minimalRL/blob/master/acer.py
# Characteristics
# 1. Discrete action space, single thread version.
# 2. Does not support trust-region updates.


      
class ACER_agent(BaseAgent):
    
    def __init__(self, obs_dim, action_dim,
                 fully_informed = True,
                 nlp = True,
                 learning_rate = 0.0002,
                 gamma         = 0.98,
                 buffer_limit  = 6000 , 
                 rollout_len   = 10   ,
                 batch_size    = 8    , # Indicates 4 sequences per mini-batch (4*rollout_len = 40 samples total)
                 c             = 1.0,   # For truncating importance sampling ratio 
                 max_sentence_length = 95,
                 episode_before_train = 100):      
            
        BaseAgent.__init__(self, action_dim, obs_dim, "ACERAgent", fully_informed, nlp)            
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer_limit = buffer_limit
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.c = c
        self.max_sentence_length = max_sentence_length
        self.episode_before_train = episode_before_train
        
        self.reset()
        
    def train(self, on_policy=False):
        s,a,r,prob,done_mask,is_first = self.memory.sample(on_policy)
        
        s = s.to(self.device)
        a = a.to(self.device)
        # r = r.to(self.device)
        prob = prob.to(self.device)
        if prob.dim() != 2: prob = prob.squeeze()
        # done_mask = done_mask.to(self.device)
        
        q = self.model.q(s)
        q_a = q.gather(1,a)
        pi = self.model.pi(s, softmax_dim = 1)
        pi_a = pi.gather(1,a)
        v = (q * pi).sum(1).unsqueeze(1).detach()
        
        rho = pi.detach()/prob
        rho_a = rho.gather(1,a)
        rho_bar = rho_a.clamp(max=self.c)
        correction_coeff = (1-self.c/rho).clamp(min=0)
    
        q_ret = v[-1] * done_mask[-1]
        q_ret_lst = []
        for i in reversed(range(len(r))):
            q_ret = r[i] + self.gamma * q_ret
            q_ret_lst.append(q_ret.item())
            q_ret = rho_bar[i] * (q_ret - q_a[i]) + v[i]
            
            if is_first[i] and i!=0:
                q_ret = v[i-1] * done_mask[i-1] # When a new sequence begins, q_ret is initialized  
                
        q_ret_lst.reverse()
        q_ret = torch.tensor(q_ret_lst, dtype=torch.float, device=self.device).unsqueeze(1)
        
        loss1 = -rho_bar * torch.log(pi_a) * (q_ret - v) 
        loss2 = -correction_coeff * pi.detach() * torch.log(pi) * (q.detach()-v) # bias correction term
        loss = loss1 + loss2.sum(1) + F.smooth_l1_loss(q_a, q_ret)
        
        self.optimizer.zero_grad()
        loss.mean().backward()
        for name,param in self.model.named_parameters(): 
            if not param.grad is None: param.grad.data.clamp_(-1, 1)
        self.optimizer.step()        
            
        
    def reset(self):
        
        if not self.nlp:
            self.model = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        else:
            if self.fully_informed: k = 5
            else:                   k = 100
            # self.model = NLP_ActorCritic(self.voc_size, k, self.action_dim).to(self.device)
            self.model = NLP_ActorCritic(k, self.action_dim).to(self.device)
        
        
        
        self.memory = ReplayBuffer(self.buffer_limit, self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)    
            
        self.seq_data = []
        
        
    def act(self, state, test = False):
        state, _ = self.filter_state(state, None)
        torch.from_numpy(state).to(self.device)
        
        if test:
            return self.model.pi(torch.from_numpy(state).float().to(self.device)).argmax().item()
        
        prob = self.model.pi(torch.from_numpy(state).float().to(self.device))
        
        # assert all(prob>0), f"{prob},{state}"
        
        return Categorical(prob).sample().item()    
    
    
    def before_act(self):
        if len(self.seq_data) > self.rollout_len: self.seq_data = []
            
        
    def update(self, i, state, action, next_state, reward):
        state, next_state = self.filter_state(state, next_state)
        prob = self.model.pi(torch.from_numpy(state).float().to(self.device)).detach().cpu().numpy()
        
        self.seq_data.append((state, 
                              action, 
                              reward/100.0, 
                              prob, 
                              next_state is None))
    
        if len(self.seq_data) == self.rollout_len:
            self.memory.put(self.seq_data.copy())
            if self.memory.size()>self.episode_before_train:
                self.train(on_policy=True)
                self.train()    
    
            
    
    
    
# import torch
# agent.model.pi(torch.tensor(agent.tokenize(NLP_env.reset())).float())
# agent.model.pi(torch.tensor(NNLP_env.reset()).float())
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    