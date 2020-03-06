import gym,os
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from nlp2020.agents.base_agent import BaseAgent


# https://github.com/seungeunrho/minimalRL/blob/master/acer.py
# Characteristics
# 1. Discrete action space, single thread version.
# 2. Does not support trust-region updates.


class ReplayBuffer():
    def __init__(self, buffer_limit, batch_size):
        self.batch_size = batch_size
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, seq_data):
        self.buffer.append(seq_data)
    
    def sample(self, on_policy=False):
        if on_policy:
            mini_batch = [self.buffer[-1]]
        else:
            mini_batch = random.sample(self.buffer, self.batch_size)

        s_lst, a_lst, r_lst, prob_lst, done_lst, is_first_lst = [], [], [], [], [], []
        for seq in mini_batch:
            is_first = True  # Flag for indicating whether the transition is the first item from a sequence
            for transition in seq:
                s, a, r, prob, done = transition

                s_lst.append(s)
                a_lst.append([a])
                r_lst.append(r)
                prob_lst.append(prob)
                done_mask = 0.0 if done else 1.0
                done_lst.append(done_mask)
                is_first_lst.append(is_first)
                is_first = False

        s,a,r,prob,done_mask,is_first = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                        r_lst, torch.tensor(prob_lst, dtype=torch.float), done_lst, \
                                        is_first_lst
        return s,a,r,prob,done_mask,is_first
    
    def size(self):
        return len(self.buffer)
      
    
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared
        self.fc1 = nn.Linear(obs_dim,256)
        self.fc2 = nn.Linear(256,128)
        
        # Pi
        self.fc_pi1 = nn.Linear(128,64)
        self.fc_pi2 = nn.Linear(64,action_dim)
   
        # Q     
        self.fc_q1 = nn.Linear(128,64)
        self.fc_q2 = nn.Linear(64,action_dim)
        
    
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = self.fc_pi1(x)
        x = self.fc_pi2(x)
        
        return F.softmax(x, dim=softmax_dim)
    
    def q(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))   
        
        x = torch.tanh(self.fc_q1(x))        
        x = torch.tanh(self.fc_q2(x))        
        
        return x
      
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
                 max_sentence_length = 100,
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
        self.optimizer.step()        
            
        
    def reset(self):
        self.model = ActorCritic(self.obs_dim, self.action_dim).to(self.device)
        
        
        self.memory = ReplayBuffer(self.buffer_limit, self.batch_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)    
            
        self.seq_data = []
        
        
    def act(self, state, test = False):
        if test:
            return self.model.pi(torch.from_numpy(state).float()).argmax().item()
        
        prob = self.model.pi(torch.from_numpy(state).float().to(self.device))
        return Categorical(prob).sample().item()    
    
    
    def before_act(self):
        if len(self.seq_data) > self.rollout_len: self.seq_data = []
            
        
    def update(self, i, state, action, next_state, reward):
        prob = self.model.pi(torch.from_numpy(state).float().to(self.device)).detach().cpu().numpy()
        
        self.seq_data.append((state, 
                              action, 
                              reward/100.0, 
                              prob, 
                              next_state is None))
    
        if len(self.seq_data) == self.rollout_len:
            self.memory.put(self.seq_data.copy())
            if self.memory.size()>500:
                self.train(on_policy=True)
                self.train()    
    
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    