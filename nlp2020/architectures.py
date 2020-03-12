import torch.nn as nn
import torch
import torch.nn.functional as F
import collections, random
import numpy as np
import re

from transformers import BertForSequenceClassification, AdamW, BertConfig

from nltk.corpus import stopwords

try: stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    stopwords.words('english')



class NLP_NN_EASY(nn.Module):
    def __init__(self, vocab_size, k):
        super().__init__()
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        
        self.fc = nn.Linear(vocab_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, k)
        
        self.do = nn.Dropout(0.1)
        self.sm = torch.nn.Softmax(dim=1)
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, x):

        if x.dim() == 1: x = x.view(1,-1)
        
        
        # x = self.embedding(x)
        x = torch.tanh(self.do(self.fc(x)))
        x = torch.tanh(self.do(self.fc1(x)))
        x = torch.tanh(self.do(self.fc2(x)))
        
        return self.sm(x)



    
class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        nn.Module.__init__(self)

        self.hl1 = nn.Linear(inputs, 256)
        self.hl2 = nn.Linear(256, 128)
        self.hl3 = nn.Linear(128, 64)
        self.hl4 = nn.Linear(64, outputs)
        self.distribution = torch.distributions.Categorical
        
        # if torch.cuda.is_available(): self.cuda()
        
    def forward(self, x):
        x = self.hl1(x)
        x = torch.tanh(self.hl2(x))
        x = torch.tanh(self.hl3(x))
        x = self.hl4(x)
        
        return x


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        nn.Module.__init__(self)
        
        # Shared
        self.fc1 = nn.Linear(obs_dim,256)
        self.fc2 = nn.Linear(256,128)
        
        # Pi
        self.fc_pi1 = nn.Linear(128,64)
        self.fc_pi2 = nn.Linear(64,action_dim)
   
        # Q     
        self.fc_q1 = nn.Linear(128,64)
        self.fc_q2 = nn.Linear(64,action_dim)
        
    
    def pi(self, x, softmax_dim = 1):
        if x.dim() == 1: x = x.view(1,-1)
        
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

# class NLP_ActorCritic(nn.Module):

#     def __init__(self, k, action_dim):
#         nn.Module.__init__(self)   
#         # self.NLP = NLP_NN_EASY(vocab_dim, k)
#         self.NLP = NLP_NN(k)
#         self.RL  = ActorCritic(k, action_dim)
    
#     def pi(self, x, softmax_dim = 0): 
#         if x.dim() != 2: 
#             x = x.squeeze()
#         return self.RL.pi(self.NLP(x))
#     def q(self, x):                   
#         if x.dim() != 2: 
#             x = x.squeeze()
#         return self.RL.q(self.NLP(x))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = collections.namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def push(self, state, action, next_state, reward):
        """Saves a transition."""
        if len(self.memory) < self.capacity: self.memory.append(None)
        self.memory[self.position] = self.transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self):            return len(self.memory)



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


        s,a,r,prob,done_mask,is_first = s_lst, torch.tensor(a_lst), \
                                        r_lst, torch.tensor(prob_lst, dtype=torch.float), done_lst, \
                                        is_first_lst
        return s,a,r,prob,done_mask,is_first
    
    def size(self):
        return len(self.buffer)


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sm = torch.nn.Softmax(dim=1)
        
    def forward(self, text):
        text = text.permute(1, 0) # Do you really need to permute?
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.sm(self.fc(cat))


class NLP_ActorCritic(nn.Module):

    def __init__(self, k, action_dim,vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        nn.Module.__init__(self)   
        # self.NLP = NLP_NN_EASY(vocab_dim, k)
        self.NLP = CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx)
        self.RL  = ActorCritic(k, action_dim)
    
    def pi(self, x, softmax_dim = 0): 
        if x.dim() != 2: 
            x = x.squeeze()
        return self.RL.pi(self.NLP(x))
    def q(self, x):                   
        if x.dim() != 2: 
            x = x.squeeze()
        return self.RL.q(self.NLP(x))

























