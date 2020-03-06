import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, math

from nlp2020.agents.base_agent import BaseAgent
from nlp2020.architectures import NLP_NN, DQN, ReplayMemory


class DQN_agent(BaseAgent):
    
    def __init__(self, obs_dim, action_dim,
                 fully_informed = True,
                 nlp = False,
                 batch_size = 64,
                 gamma = 0.999,
                 eps_start = 0.9,
                 eps_end = 0.01,
                 eps_decay = 200,
                 target_update = 100,
                 buffer_size = 10000,
                 max_sentence_length = 201
                 ):
        
        BaseAgent.__init__(self, action_dim, obs_dim, "DQNAgent", fully_informed, nlp)        
        
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
        
        if len(self.memory) < self.batch_size: return
        transitions = self.memory.sample(self.batch_size)
        batch = self.memory.transition(*zip(*transitions))
    
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.stack([s for s in batch.next_state
                                                    if s is not None], dim=0).squeeze()
        # state_batch = torch.stack(torch.tensor(batch.state).to(self.device),dim=0).squeeze()
        state_batch = torch.tensor(batch.state).to(self.device).squeeze()
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
    
    
        state_action_values = self.model(state_batch).gather(1, action_batch.view(-1,1))
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
    
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters(): param.grad.data.clamp_(-1, 1)
        self.optimizer.step()        
            
        
    def update(self, i, state, action, next_state, reward):
        reward = torch.as_tensor([reward], device=self.device)
        action = torch.as_tensor([action], dtype = torch.long, device = self.device)
        state, next_state = self.filter_state(state, next_state)

        if not next_state is None:
            next_state = torch.as_tensor(next_state, dtype = torch.float, device = self.device)
        
        self.memory.push(state, action, next_state, reward)

        # Perform one step of the optimization (on the target network)
        self.optimize_model()

        if i>0 and i % self.target_update == 0:
            self.target_net.load_state_dict(self.model.state_dict())
    

    def is_greedy_step(self):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        return sample > eps_threshold


    def act(self, state, test = False):
        state, _ = self.filter_state(state, None)
        self.steps_done += 1
        
        if self.is_greedy_step() or test:
            with torch.no_grad(): 
                return self.model(torch.from_numpy(state).to(self.device).float()).argmax().item()
        else:                     return random.randrange(self.n_actions)
            
        
    def reset(self):
        self.steps_done = 0

        if not self.nlp:
            self.model = DQN(self.obs_dim, self.action_dim).to(self.device)
            self.target_net = DQN(self.obs_dim, self.action_dim).to(self.device)
        else:
            if self.fully_informed: k = 5
            else:                   k = 100
            
            self.model = nn.Sequential(NLP_NN(k), DQN(k,self.action_dim)).to(self.device)
            self.target_net = nn.Sequential(NLP_NN(k), DQN(k,self.action_dim)).to(self.device)
            
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(self.model.parameters())
        self.memory = ReplayMemory(self.buffer_size)


    def load_model(self):
        BaseAgent.load_model(self)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()

















