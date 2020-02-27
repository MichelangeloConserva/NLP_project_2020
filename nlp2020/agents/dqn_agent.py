import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import random, math, os

from collections import namedtuple
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig

from nlp2020.agents.base_agent import BaseAgent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# pytorch doc



class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
    
class DQN(nn.Module):

    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()

        self.hl1 = nn.Linear(inputs, 256)
        self.hl2 = nn.Linear(256, 128)
        self.hl3 = nn.Linear(128, 64)
        self.hl4 = nn.Linear(64, outputs)
        self.distribution = torch.distributions.Categorical
        
    def forward(self, x):
        x = self.hl1(x)
        x = torch.tanh(self.hl2(x))
        x = torch.tanh(self.hl3(x))
        x = self.hl4(x)
        
        return x


class NLP_NN(nn.Module):

    
    def tokenize(x):
        # TODO : implement tokenizer
        return torch.tensor([])
    
    
    def __init__(self, outputs):
        super(NLP_NN, self).__init__()
        
        self.outputs = outputs
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                                       do_lower_case=True)

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
            num_labels = 5, # The number of output labels--2 for binary classification.
                            # You can increase this for multi-class tasks.   
            output_attentions = False, # Whether the model returns attentions weights.
            output_hidden_states = False, # Whether the model returns all hidden-states.
        )
        
        # TODO : activate back for training
        # self.model.cuda()

        self.sm = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.model(x)[0]
        
        return self.sm(x)






class DQN_agent(BaseAgent):
    
    def __init__(self, obs_dim, action_dim,
                 fully_informed = True,
                 nlp = False,
                 batch_size = 128,
                 gamma = 0.999,
                 eps_start = 0.9,
                 eps_end = 0.01,
                 eps_decay = 200,
                 target_update = 100,
                 buffer_size = 10000,
                 max_sentence_length = 201
                 ):
        
        BaseAgent.__init__(self, action_dim, obs_dim, "DQNAgent", fully_informed, nlp)        
        
        # TODO : remove False
        self.device = torch.device("cuda" if torch.cuda.is_available() and False else "cpu")
        self.n_actions = action_dim
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.buffer_size = buffer_size
        self.max_sentence_length = max_sentence_length
        
        # Create the NNs
        self.reset()
    
    
    def optimize_model(self):
        
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None], dim=0)
        state_batch = torch.stack(batch.state,dim=0).squeeze()
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.view(-1,1))
    
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()        
            
        
    def update(self, i, state, action, next_state, reward):
        reward = torch.as_tensor([reward], device=self.device)
        action = torch.as_tensor([action], dtype = torch.long, device = self.device)
        
        
        if not self.nlp:  
            state = torch.tensor(state, dtype = torch.float, device = self.device)
            if not next_state is None:
                next_state = torch.as_tensor(next_state, dtype = torch.float, device = self.device)
            else:
                next_state = torch.zeros(self.max_sentence_length, dtype = torch.long)        
        else:             
            state = self.tokenize(state)     
            if not next_state is None:
                    next_state = self.tokenize(next_state)
            else:
                next_state = torch.zeros(self.max_sentence_length, dtype = torch.long)
                    
                    
                    
        if not next_state is None:
            next_state = torch.as_tensor(next_state, dtype = torch.float, device = self.device)
        
        self.memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        if i>0 and i % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    

    def is_greedy_step(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        return sample > eps_threshold


    def act(self, state, test = False):
        
        if not self.nlp:
            state = torch.tensor(state, dtype = torch.float, device = self.device)
        else:
            state = self.tokenize(state)
        
        self.steps_done += 1
        
        if self.is_greedy_step() or test:
            with torch.no_grad(): return self.policy_net(state).argmax().item()
        else:                     return random.randrange(self.n_actions)
            
        
    def reset(self):
        self.steps_done = 0

        if not self.nlp:
            self.policy_net = DQN(self.obs_dim, self.action_dim).to(self.device)
            self.target_net = DQN(self.obs_dim, self.action_dim).to(self.device)
        
        
        else:
            
            if self.fully_informed: k = 5
            else:                   k = 100
            
            self.policy_net = nn.Sequential(NLP_NN(k), DQN(k,self.action_dim))
            self.target_net = nn.Sequential(NLP_NN(k), DQN(k,self.action_dim))
            
            self.tokenizer = list(self.policy_net.children())[0].tokenizer
        
            
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(self.buffer_size)


        
        

    def save_model(self, save_dir = "./logs_custom/DQN/"):
        BaseAgent.save_model(self, save_dir, self.policy_net)

    def load_model(self, load_file = "./logs_custom/DQN/model"):
        self.policy_net = BaseAgent.load_model(self, load_file, "policy_net")
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()



    def tokenize(self, sentence):
        token = [self.tokenizer.encode(sentence, add_special_tokens = True)]
        token = pad_sequences(token, maxlen=self.max_sentence_length, 
                              dtype="long", value=0, truncating="post", padding="post")
        
        return torch.tensor(token, device = self.device).long()
















