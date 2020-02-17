import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple
import random, math

from nlp2020.agents.base_agent import BaseAgent

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



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


class DQN_agent(BaseAgent):
    
    def __init__(self, obs_dim, action_dim,
                 batch_size = 128,
                 gamma = 0.999,
                 eps_start = 0.9,
                 eps_end = 0.01,
                 eps_decay = 200,
                 target_update = 100
                 ):
        
        BaseAgent.__init__(self, 
                           action_dim, 
                           obs_dim, 
                           "DQNAgent")        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = action_dim
        
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update

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
        state_batch = torch.stack(batch.state,dim=0)
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
        state = torch.as_tensor(state, dtype = torch.float, device = self.device)
        if not next_state is None:
            next_state = torch.as_tensor(next_state, dtype = torch.float, device = self.device)
        
        
        self.memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        if i>0 and i % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    

    def act(self, state):
        state = torch.tensor(state, dtype = torch.float, device = self.device)
        
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # return self.policy_net(state).max(1)[1].view(1, 1)
                return self.policy_net(state).argmax().item()
        else:
            return random.randrange(self.n_actions)


    def reset(self):
        
        self.steps_done = 0
        
        self.policy_net = DQN(self.obs_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.obs_dim, self.action_dim).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)




