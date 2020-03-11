import numpy as np
import matplotlib.pyplot as plt
import re
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from nltk.corpus import stopwords
from scipy.stats import invgamma
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nlp2020.dung_descr_score import dungeon_description_generator

try: stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    stopwords.words('english')
    
    
def smooth(array, smoothing_horizon=100., initial_value=0.):
  """Smoothing function for plotting. Credit to Deep Mind Lectures at UCL"""
  smoothed_array = []
  value = initial_value
  b = 1./smoothing_horizon
  m = 1.
  for x in array:
    m *= 1. - b
    lr = b/(1 - m)
    value += lr*(x - value)
    smoothed_array.append(value)
  return np.array(smoothed_array)


def multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials):
    # Multi bars plot
    spacing = np.linspace(-1,1, len(algs))
    width = spacing[1] - spacing[0]
    missions = np.arange((n_mission_per_episode+1)*4, step = 4)
    for (i,(_,(agent,_,_,_,col,_))) in enumerate(algs.items()):
        c = Counter(test_trials[agent.name])
        counts = np.zeros(n_mission_per_episode+1)
        for k,v in c.items(): counts[k] = v/n_test_trials

        assert round(sum(counts),5) == 1
        plt.bar(missions + spacing[i], 
                counts, width, label = agent.name, color = col, edgecolor="black")
        
    plt.xlabel("Consecutive mission, i.e. length of the episode")
    plt.xticks(missions,range(n_mission_per_episode+1))
    plt.legend()
    
def tokenize(sentence,max_sentence_length=95):
    
    # Remove punctuations and numbers
    sentence =  re.sub('[^a-zA-Z]', ' ', sentence)[:-1].lower()

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence).split(" ")        
    
    sentence = [word for word in sentence if (word not in stopwords.words('english'))]
    
    # token = pad_sequences(sentence, maxlen=max_sentence_length, 
    #                       dtype="long", value=0, truncating="post", padding="post")
    token = sentence + [""] * (max_sentence_length - len(sentence))
    
    return token

def onehot_to_int(v): return v.index(1)
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataset(n = 2000):
    train_x = []
    train_y_temp = []
    train_y = []
    for i in range(n):
      description, label, _ = dungeon_description_generator()
      train_x.append(description)
      train_y_temp.append(label)
    for i in train_y_temp: train_y.append(onehot_to_int(i.tolist()))
    
    val_x = []
    val_y_temp = []
    val_y = []
    for i in range(n):
      description, label, _ = dungeon_description_generator()
      val_x.append(description)
      val_y_temp.append(label)
    for i in val_y_temp: val_y.append(onehot_to_int(i.tolist()))
    
    test_x = []
    test_y_temp = []
    test_y = []
    for i in range(n):
      description, label, _ = dungeon_description_generator()
      test_x.append(description)
      test_y_temp.append(label)
    for i in test_y_temp: test_y.append(onehot_to_int(i.tolist()))

    return train_x, val_x, test_x, train_y, val_y, test_y

def ListToTorchtext(train_x, val_x, test_x, train_y, val_y, test_y, datafields):
  train = []
  for i,line in enumerate(train_x):
    doc = line.split()
    train.append(torchtext.data.Example.fromlist([doc, train_y[i]], datafields))
  val = []
  for i,line in enumerate(val_x):
    doc = line.split()
    val.append(torchtext.data.Example.fromlist([doc, val_y[i]], datafields))
  test = []
  for i,line in enumerate(test_x):
    doc = line.split()
    test.append(torchtext.data.Example.fromlist([doc, test_y[i]], datafields))
  return torchtext.data.Dataset(train, datafields), torchtext.data.Dataset(val, datafields), torchtext.data.Dataset(test, datafields)


def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])





class ContextualDataset(object):
  """The buffer is able to append new data, and sample random minibatches."""

  def __init__(self, context_dim, num_actions, buffer_s=-1, intercept=False):
    """Creates a ContextualDataset object.
    The data is stored in attributes: contexts and rewards.
    The sequence of taken actions are stored in attribute actions.
    Args:
      context_dim: Dimension of the contexts.
      num_actions: Number of arms for the multi-armed bandit.
      buffer_s: Size of buffer for training. Only last buffer_s will be
        returned as minibatch. If buffer_s = -1, all data will be used.
      intercept: If True, it adds a constant (1.0) dimension to each context X,
        at the end.
    """

    self._context_dim = context_dim
    self._num_actions = num_actions
    self._contexts = None
    self._rewards = None
    self.actions = []
    self.buffer_s = buffer_s
    self.intercept = intercept

  def add(self, context, action, reward):
    """Adds a new triplet (context, action, reward) to the dataset.
    The reward for the actions that weren't played is assumed to be zero.
    Args:
      context: A d-dimensional vector with the context.
      action: Integer between 0 and k-1 representing the chosen arm.
      reward: Real number representing the reward for the (context, action).
    """

    if self.intercept:
      c = np.array(context[:])
      c = np.append(c, 1.0).reshape((1, self.context_dim + 1))
    else:
      c = np.array(context[:]).reshape((1, self.context_dim))

    if self.contexts is None:
      self.contexts = c
    else:
      self.contexts = np.vstack((self.contexts, c))

    r = np.zeros((1, self.num_actions))
    r[0, action] = reward
    if self.rewards is None:
      self.rewards = r
    else:
      self.rewards = np.vstack((self.rewards, r))

    self.actions.append(action)

  def replace_data(self, contexts=None, actions=None, rewards=None):
    if contexts is not None:
      self.contexts = contexts
    if actions is not None:
      self.actions = actions
    if rewards is not None:
      self.rewards = rewards

  def get_batch(self, batch_size):
    """Returns a random minibatch of (contexts, rewards) with batch_size."""
    n, _ = self.contexts.shape
    if self.buffer_s == -1:
      # use all the data
      ind = np.random.choice(range(n), batch_size)
    else:
      # use only buffer (last buffer_s observations)
      ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)
    return self.contexts[ind, :], self.rewards[ind, :]

  def get_data(self, action):
    """Returns all (context, reward) where the action was played."""
    n, _ = self.contexts.shape
    ind = np.array([i for i in range(n) if self.actions[i] == action])
    return self.contexts[ind, :], self.rewards[ind, action]

  def get_data_with_weights(self):
    """Returns all observations with one-hot weights for actions."""
    weights = np.zeros((self.contexts.shape[0], self.num_actions))
    a_ind = np.array([(i, val) for i, val in enumerate(self.actions)])
    weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
    return self.contexts, self.rewards, weights

  def get_batch_with_weights(self, batch_size):
    """Returns a random mini-batch with one-hot weights for actions."""
    n, _ = self.contexts.shape
    if self.buffer_s == -1:
      # use all the data
      ind = np.random.choice(range(n), batch_size)
    else:
      # use only buffer (last buffer_s obs)
      ind = np.random.choice(range(max(0, n - self.buffer_s), n), batch_size)

    weights = np.zeros((batch_size, self.num_actions))
    sampled_actions = np.array(self.actions)[ind]
    a_ind = np.array([(i, val) for i, val in enumerate(sampled_actions)])
    weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
    return self.contexts[ind, :], self.rewards[ind, :], weights

  def num_points(self, f=None):
    """Returns number of points in the buffer (after applying function f)."""
    if f is not None:
      return f(self.contexts.shape[0])
    return self.contexts.shape[0]

  @property
  def context_dim(self):
    return self._context_dim

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def contexts(self):
    return self._contexts

  @contexts.setter
  def contexts(self, value):
    self._contexts = value

  @property
  def actions(self):
    return self._actions

  @actions.setter
  def actions(self, value):
    self._actions = value

  @property
  def rewards(self):
    return self._rewards

  @rewards.setter
  def rewards(self, value):
    self._rewards = value





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
        
    def forward(self, text):
        text = text.permute(1, 0)
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim = 1))
            
        return self.fc(cat)



class NeuralBanditModel(nn.Module):
    """Implements a neural network for bandit problems."""

    def __init__(self, 
               vocab_size, 
               embedding_dim, 
               n_filters, 
               filter_sizes, 
               dropout, 
               pad_idx  ,
               init_s,
               context_dim,
               num_action,
               name
               ):
        nn.Module.__init__(self)
    
        self.name = name
        
        # TODO : this should be a list of sizes for each layer
        self.init_s = init_s
        
        self.verbose = False
        self.times_trained = 0
    
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, 256)
        self.fc2 = nn.Linear(256, 128)
        self.nn = nn.Linear(128, 64)
        self.pred = nn.Linear(64, num_action)
    
        self.fc1.weight.data.uniform_(-init_s, init_s)
        self.fc2.weight.data.uniform_(-init_s, init_s)
        self.nn.weight.data.uniform_(-init_s, init_s)
        self.pred.weight.data.uniform_(-init_s, init_s)

        self.dropout = nn.Dropout(dropout)    

    def latent(self,text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        x = self.dropout(torch.cat(pooled, dim = 1))        
        
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.dropout(torch.tanh(self.nn(x)))
        return x
    
    def forward(self, x):
        x = self.latent(x)
        return self.pred(x)

    
class BanditAlgorithm():
  """A bandit algorithm must be able to do two basic operations.
  1. Choose an action given a context.
  2. Update its internal model given a triple (context, played action, reward).
  """

  def action(self, context):
    pass

  def update(self, context, action, reward):
    pass
    

class NeuralLinearPosteriorSampling(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, 
                 vocab_size,
                 embedding_dim,
                 n_filters,
                 dropout,
                 pad_idx,   
                 filter_sizes,
                num_actions=7,
                context_dim=95,
                init_scale=0.3,
                activation=torch.relu,
                batch_size=512,
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
                lr_decay_rate=0.5,
                training_freq=1,
                training_freq_network=50,
                training_epochs=100,
                a0=6,
                b0=6,
                lambda_prior=0.25,
                optimizer='RMS',
                device = "cuda"):

    self.name = name
    self.num_actions=num_actions
    self.context_dim = context_dim
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
    self.optimizer=optimizer
    
    self.times_trained = 0
    
    self.bnn = NeuralBanditModel(vocab_size,
                                 embedding_dim,
                                 n_filters,
                                 filter_sizes = filter_sizes,
                                 dropout = dropout,
                                 pad_idx = pad_idx,                                
                                 init_s = init_scale,
                                 context_dim = 95,
                                 num_action = 7,
                                 name = '{}-bnn'.format(name))
    self.device = device
    self.bnn.to(device)
    self.optimizer = torch.optim.RMSprop(self.bnn.parameters(), lr = initial_lr)
    self.lr_scheduler = ReduceLROnPlateau(self.optimizer,  
            factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=0.5e-6, verbose=1)
    
    
    self.latent_dim = self.bnn.pred.in_features    
    
    self.t = 0
    self.optimizer_n = optimizer

    self.num_epochs = self.training_epochs
    self.data_h = ContextualDataset(self.context_dim,
                                    self.num_actions,
                                    intercept=False)
    self.latent_h = ContextualDataset(self.latent_dim,
                                      self.num_actions,
                                      intercept=False)

    # Gaussian prior for each beta_i
    self._lambda_prior = self.lambda_prior

    self.mu = [
        np.zeros(self.latent_dim)
        for _ in range(self.num_actions)
    ]

    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.latent_dim)
                for _ in range(self.num_actions)]

    self.precision = [
        self.lambda_prior * np.eye(self.latent_dim)
        for _ in range(self.num_actions)
    ]

    # Inverse Gamma prior for each sigma2_i
    self.a0 = self.a0
    self.b0 = self.b0

    self.a = [self.a0 for _ in range(self.num_actions)]
    self.b = [self.b0 for _ in range(self.num_actions)]

    # Regression and NN Update Frequency
    self.update_freq_lr = self.training_freq
    self.update_freq_nn = self.training_freq_network



  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.num_actions * self.initial_pulls:
      return self.t % self.num_actions

    # Sample sigma2, and beta conditional on sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i])
        for i in range(self.num_actions)
    ]

    try:
      beta_s = [
          np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
          for i in range(self.num_actions)
      ]
    except np.linalg.LinAlgError as e:
      # Sampling could fail if covariance is not positive definite
      print('Exception when sampling for {}.'.format(self.name))
      print('Details: {} | {}.'.format(e.message, e.args))
      d = self.latent_dim
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for i in range(self.num_actions)
      ]

    # Compute last-layer representation for the current context
    with torch.no_grad():
        context = torch.from_numpy(context).to(self.device).long()
        z_context = self.bnn.latent(context).cpu().numpy()

    # Apply Thompson Sampling to last-layer representation
    vals = [
        np.dot(beta_s[i], z_context.T) for i in range(self.num_actions)
    ]
    
    return np.argmax(vals)


  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1
    self.data_h.add(context, action, reward)
    
    with torch.no_grad():
        context = torch.from_numpy(context).to(self.device).long()
        z_context = self.bnn.latent(context).cpu().numpy()
    
    self.latent_h.add(z_context, action, reward)

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:

        
      # TODO : learing rate scheduler
      # if self.reset_lr:
      #   self.bnn.assign_lr()
        

      for step in range(self.num_epochs):
        x, y, w = self.data_h.get_batch_with_weights(self.batch_size)
        
        x = torch.tensor(x, device = self.device).long()
        y = torch.tensor(y, device = self.device).long()
        w = torch.tensor(w, device = self.device).long()
        
        y_pred = self.bnn(x)
        
        self.loss = (y_pred - y)**2
        self.loss = self.loss * w
        self.loss =  self.loss.sum() / self.batch_size
        
        # print(self.loss, self.loss.shape)

        
        self.optimizer.zero_grad()
        self.loss.backward()
        for name,param in self.bnn.named_parameters(): 
            if not param.grad is None: param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.times_trained += 1

      self.lr_scheduler.step(self.loss)
        
      self.loop.postfix = "{} | step: {}, lr: {}, loss: {}".format(
                self.name, step, round(self.optimizer.param_groups[0]['lr'],5), round(self.loss.item(),2))


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



class ContextualBandit(object):
  """Implements a Contextual Bandit with d-dimensional contexts and k arms."""

  def __init__(self, context_dim, num_actions):
    """Creates a contextual bandit object.
    Args:
      context_dim: Dimension of the contexts.
      num_actions: Number of arms for the multi-armed bandit.
    """

    self._context_dim = context_dim
    self._num_actions = num_actions

  def feed_data(self, data):
    """Feeds the data (contexts + rewards) to the bandit object.
    Args:
      data: Numpy array with shape [n, d+k], where n is the number of contexts,
        d is the dimension of each context, and k the number of arms (rewards).
    Raises:
      ValueError: when data dimensions do not correspond to the object values.
    """

    if data.shape[1] != self.context_dim + self.num_actions:
      raise ValueError('Data dimensions do not match.')

    self._number_contexts = data.shape[0]
    self.data = data
    self.order = range(self.number_contexts)

  def reset(self):
    """Randomly shuffle the order of the contexts to deliver."""
    self.order = np.random.permutation(self.number_contexts)

  def context(self, number):
    """Returns the number-th context."""
    return self.data[self.order[number]][:self.context_dim]

  def reward(self, number, action):
    """Returns the reward for the number-th context and action."""
    return self.data[self.order[number]][self.context_dim + action]

  def optimal(self, number):
    """Returns the optimal action (in hindsight) for the number-th context."""
    return np.argmax(self.data[self.order[number]][self.context_dim:])

  @property
  def context_dim(self):
    return self._context_dim

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def number_contexts(self):
    return self._number_contexts










def get_vocab():
    
    import numpy as np
    import torch
    import torchtext
    import random
    import matplotlib.pyplot as plt
    
    from tqdm import tqdm
    from torchtext import data
    
    from nlp2020.dung_descr_score import dungeon_description_generator
    from nlp2020.utils import tokenize, ListToTorchtext, ContextualDataset, NeuralBanditModel,NeuralLinearPosteriorSampling
    from nlp2020.utils import ContextualBandit, multi_bar_plot, smooth
    
    
    num_dung = 5
    num_weap = 7
    
    reward_win = 5
    reward_die = -10
    
    
    # =============================================================================
    # Worst way possible to get the dict
    # =============================================================================
    
    N = 3000
    x_train = []
    y_train = []
    for n in range(N):
        context, dung_identifier, probs_per_weapon = dungeon_description_generator()
    
        x_train.append(context)
        y_train.append(
        [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
        )
        
    x_test = []
    y_test = []
    for n in range(N):
        context, dung_identifier, probs_per_weapon = dungeon_description_generator()
    
        x_test.append(context)
        y_test.append(
        [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
        )
        
    x_val = []
    y_val = []
    for n in range(N):
        context, dung_identifier, probs_per_weapon = dungeon_description_generator()
    
        x_val.append(context)
        y_val.append(
        [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
        )
    
    
    TEXT = data.Field(tokenize = tokenize)
    LABEL = data.RawField()
    datafields = [('text', TEXT), ('label', LABEL)]
    
    TrainData, ValData, TestData = ListToTorchtext(x_train, x_val, x_test, y_train, y_val, y_test, datafields)
    
    TEXT.build_vocab(TrainData)
    # print(len(TEXT.vocab))s

    return TEXT



















