import torch.nn as nn
import torch
import torch.nn.functional as F

from nltk.corpus import stopwords

try: stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    stopwords.words('english')

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
        
        if output_dim != 5: self.sm = torch.tanh
        else:                self.sm = torch.nn.Softmax(dim=1)
        
    def forward(self, text):
        text = text.permute(1, 0) # Do you really need to permute?
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.leaky_relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.sm(self.fc(cat))

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, dp_rl):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(obs_dim,256); self.fc2 = nn.Linear(256,128)          # Shared
        self.fc_pi1 = nn.Linear(128,64);   self.fc_pi2 = nn.Linear(64,action_dim) # Pi
        self.fc_q1 = nn.Linear(128,64);    self.fc_q2 = nn.Linear(64,action_dim)  # Q
        self.dp = nn.Dropout(dp_rl)
    
    def shared(self, x):
        return self.dp(F.leaky_relu(self.fc2(self.dp(F.leaky_relu(self.fc1(x))))))
    
    def pi(self, x, softmax_dim = 1):
        x = self.shared(x)
        return F.softmax(self.fc_pi2(self.fc_pi1(x)), dim=softmax_dim)
    
    def q(self, x): return self.fc_q2(self.fc_q1(self.shared(x)))

# SMALLER ARCHITECTURE IS NOT ENOUGH
# class ActorCritic(nn.Module):
    
#     def __init__(self, obs_dim, action_dim, dp_rl):
#         nn.Module.__init__(self)
#         self.fc1 = nn.Linear(obs_dim,32); # Shared
#         self.fc_pi1 = nn.Linear(32,action_dim) # Pi
#         self.fc_q1 = nn.Linear(32,action_dim)  # Q
#         self.dp = nn.Dropout(dp_rl)
    
#     def shared(self, x):
#         return self.dp(F.leaky_relu(self.fc1(x)))
    
#     def pi(self, x, softmax_dim = 1):
#         x = self.shared(x)
#         return F.softmax(self.fc_pi1(x), dim=softmax_dim)
    
#     def q(self, x): return self.fc_q1(self.shared(x))


class NLP_ActorCritic(nn.Module):

    def __init__(self, k, action_dim,vocab_size, embedding_dim, n_filters, 
                 filter_sizes, output_dim, dropout, pad_idx, dp_rl = 0.1):
        nn.Module.__init__(self)   
        # self.NLP = NLP_NN_EASY(vocab_dim, k)
        self.NLP = CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx)
        self.RL  = ActorCritic(k, action_dim, dp_rl)
    
    def pi(self, x, softmax_dim = 0): 
        if x.dim() != 2: x = x.squeeze()
        return self.RL.pi(self.NLP(x))
    def q(self, x):                   
        if x.dim() != 2: x = x.squeeze()
        return self.RL.q(self.NLP(x))

