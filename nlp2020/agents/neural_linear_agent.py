import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, math
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.stats import invgamma

from nlp2020.agents.base_agent import BaseAgent
from nlp2020.architectures import NLP_NN_EASY, DQN, ReplayMemory, CNN
from nlp2020.utils import ContextualBandit, NeuralBanditModel, ContextualDataset



class NLB(BaseAgent):
    
    def __init__(self, obs_dim, action_dim,
                vocab_size, embedding_dim, n_filters, filter_sizes, 
                dropout, pad_idx,
                TEXT,
                fully_informed = True,
                nlp = False,
                gamma = 0.999,
                eps_start = 0.9,
                eps_end = 0.01,
                eps_decay = 200,
                target_update = 100,
                buffer_size = 10000,
                max_sentence_length = 95,
                init_scale=0.3,
                activation=torch.relu,
                batch_size=128,
                activate_decay=True,
                initial_lr=0.5,
                min_lr=0.0001,
                lr_steps=10e3,
                lr_gamma=.999,
                max_grad_norm=5.0,
                show_training=True,
                freq_summary=1000,
                buffer_s=-1,
                initial_pulls=2,
                reset_lr=True,
                lr_decay_rate=0.01,
                training_freq=1,
                training_freq_network=25,
                training_epochs=100,
                a0=6,
                b0=6,
                lambda_prior=0.25,
                num_epochs = 50
                 ):
        
        BaseAgent.__init__(self, action_dim, obs_dim, "NLBAgent", fully_informed, nlp)        
        
        self.n_actions = action_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_sentence_length = max_sentence_length
        self.vocab_size = vocab_size
        self.embedding_dim  = embedding_dim
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes
        self.dropout = dropout
        self.pad_idx = pad_idx     
        self.TEXT = TEXT
        self.init_scale=init_scale
        self.activation=activation
        self.batch_size=batch_size
        self.activate_decay=activate_decay
        self.initial_lr=initial_lr
        self.max_grad_norm=max_grad_norm
        self.show_training=show_training
        self.freq_summary=freq_summary
        self.buffer_s=buffer_s
        self.initial_pulls=initial_pulls
        self.reset_lr=reset_lr
        self.lr_decay_rate=lr_decay_rate
        self.training_freq=training_freq
        self.training_freq_network=training_freq_network
        self.training_epochs=training_epochs
        self.a0=a0
        self.b0=b0
        self.lambda_prior=lambda_prior
        self.num_epochs = num_epochs
        
        # Regression and NN Update Frequency
        self.update_freq_lr = self.training_freq
        self.update_freq_nn = self.training_freq_network 
    
        # Create the NNs
        self.reset()
    
    
    
    def optimize_model(self):
        pass
        
        
    def update(self, i, state, action, next_state, reward):
        state, next_state = self.filter_state(state, next_state)
        self.data_h.add(state, action, reward)
        
        
        state = torch.from_numpy(state).to(self.device)
        if state.dim() == 1: state = state.view(1,-1)           
        
        self.t += 1
        
        
        with torch.no_grad():
            z_context = self.bnn.latent(state).cpu().numpy()
        
        self.latent_h.add(z_context, action, reward)
    
        # Retrain the network on the original data (data_h)
        if self.t % self.update_freq_nn == 0:
    
            for step in range(self.num_epochs):
                x, y, w = self.data_h.get_batch_with_weights(self.batch_size)
                x = torch.tensor(x, device = self.device).long()
                y = torch.tensor(y, device = self.device).long()
                w = torch.tensor(w, device = self.device).long()
                
                y_pred = self.bnn(x)
                
                self.loss = (y_pred - y)**2
                self.loss = self.loss * w
                self.loss = self.loss.sum() / self.batch_size
                
                # print(self.loss, self.loss.shape)
                
                self.optimizer.zero_grad()
                self.loss.backward()
                for name,param in self.bnn.named_parameters(): 
                    if not param.grad is None: param.grad.data.clamp_(-1, 1)
                self.optimizer.step()
                
                self.times_trained += 1
    
            self.lr_scheduler.step(self.loss)
            
            self.loop.postfix = "lr: {}, loss: {}".format(round(self.optimizer.param_groups[0]['lr'],5), round(self.loss.item(),2))
    
    
            # Update the latent representation of every datapoint collected so far
            with torch.no_grad():
                contexts = torch.tensor(self.data_h.contexts, device = self.device).long()
                new_z = self.bnn.latent(contexts).cpu().numpy()
                self.latent_h.replace_data(contexts=new_z)
        
    
    
        # Update the Bayesian Linear Regression
        if self.t % self.update_freq_lr == 0:
    
          # Find all the actions to update
          actions_to_update = self.latent_h.actions[:-self.update_freq_lr]
    
          for action_v in np.unique(actions_to_update):
    
            # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
            z, y = self.latent_h.get_data(action_v)
    
            # The algorithm could be improved with sequential formulas (cheaper)
            s = np.dot(z.T, z)
    
            # Some terms are removed as we assume prior mu_0 = 0.
            precision_a = s + self.lambda_prior * np.eye(self.latent_dim)
            cov_a = np.linalg.inv(precision_a)
            mu_a = np.dot(cov_a, np.dot(z.T, y))
    
            # Inverse Gamma posterior update
            a_post = self.a0 + z.shape[0] / 2.0
            b_upd = 0.5 * np.dot(y.T, y)
            b_upd -= 0.5 * np.dot(mu_a.T, np.dot(precision_a, mu_a))
            b_post = self.b0 + b_upd
    
            # Store new posterior distributions
            self.mu[action_v] = mu_a
            self.cov[action_v] = cov_a
            self.precision[action_v] = precision_a
            self.a[action_v] = a_post
            self.b[action_v] = b_post
        
        
    def compute_vals(self,  state, test = False):
        state, _ = self.filter_state(state, None)
        state = torch.from_numpy(state).to(self.device)
        if state.dim() == 1: state = state.view(1,-1)           
        
        
        # Round robin until each action has been selected "initial_pulls" times
        if self.t < self.n_actions * self.initial_pulls:
          return self.t % self.n_actions
    
        # Sample sigma2, and beta conditional on sigma2
        sigma2_s = [
            self.b[i] * invgamma.rvs(self.a[i])
            for i in range(self.n_actions)
        ]
    
        try:
          beta_s = [
              np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
              for i in range(self.n_actions)
          ]
        except np.linalg.LinAlgError as e:
          # Sampling could fail if covariance is not positive definite
          print('Exception when sampling for {}.'.format(self.name))
          print('Details: {} | {}.'.format(e.message, e.args))
          d = self.latent_dim
          beta_s = [
              np.random.multivariate_normal(np.zeros((d)), np.eye(d))
              for i in range(self.n_actions)
          ]
    
        # Compute last-layer representation for the current context
        with torch.no_grad():
            z_context = self.bnn.latent(state).cpu().numpy()
    
        # Apply Thompson Sampling to last-layer representation
        vals = [
            np.dot(beta_s[i], z_context.T) for i in range(self.n_actions)
        ]
        return vals

    def act(self, state, test = False, printt = False):
        vals = self.compute_vals(state, test)
        if printt: print(vals)
        
        return np.argmax(vals)


        
    def reset(self):
        self.times_trained = 0
        self.t = 0
        
        self.bnn = NeuralBanditModel(self.vocab_size,
                                     self.embedding_dim,
                                     self.n_filters,
                                     filter_sizes = self.filter_sizes,
                                     dropout = self.dropout,
                                     pad_idx = self.pad_idx,                                
                                     init_s = self.init_scale,
                                     context_dim = 95,
                                     num_action = 7,
                                     name = '{}-bnn'.format(self.name))        

        self.bnn.to(self.device)

        self.optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr = self.initial_lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,  
                factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6, verbose=0)
    
        self.latent_dim = self.bnn.pred.in_features 
    
        # self.cmab = ContextualBandit(95, 7)
        self.data_h = ContextualDataset(95,
                                        self.n_actions,
                                        intercept=False)
        self.latent_h = ContextualDataset(self.latent_dim,
                                          self.n_actions,
                                          intercept=False)    
    
        # Gaussian prior for each beta_i
        self._lambda_prior = self.lambda_prior
    
        self.mu = [
            np.zeros(self.latent_dim)
            for _ in range(self.n_actions)
        ]
    
        self.cov = [(1.0 / self.lambda_prior) * np.eye(self.latent_dim)
                    for _ in range(self.n_actions)]
    
        self.precision = [
            self.lambda_prior * np.eye(self.latent_dim)
            for _ in range(self.n_actions)
        ]
    
        # Inverse Gamma prior for each sigma2_i
        self.a0 = self.a0
        self.b0 = self.b0
    
        self.a = [self.a0 for _ in range(self.n_actions)]
        self.b = [self.b0 for _ in range(self.n_actions)]
    
