# %%
import gym, torch, torchtext
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import collections
import random
import torch.nn.functional as F
from torch.distributions import Categorical
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm
from torchtext import data

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.utils import smooth, tokenize, count_parameters, create_iterator, categorical_accuracy, ListToTorchtext
from nlp2020.train_test_functions import train1, test1
from nlp2020.architectures import CNN, ActorCritic, NLP_ActorCritic
from nlp2020.utils import ACER_agent, create_dataset

reward_win = 1
reward_die = -1
sl_rl = True

SEED = 1234
batch_size = 256
device = torch.device("cuda")
train_iterator, _, _, LABEL, TEXT = create_iterator(device, batch_size, int(2e3))

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
vocab_size     = len(TEXT.vocab)
embedding_dim  = 128
n_filters      = 300
filter_sizes   = [2,3,4]
dropout        = 0.1
pad_idx        = TEXT.vocab.stoi[TEXT.pad_token]

low_eff = 0.1
weapon_in_dung_score = np.array([[1.,low_eff,low_eff,low_eff,low_eff,low_eff,2*low_eff],
                                 [low_eff,1.,low_eff,low_eff,low_eff,low_eff,2*low_eff],
                                 [low_eff,low_eff,1.,low_eff,low_eff,low_eff,2*low_eff],
                                 [low_eff,low_eff,low_eff,1.,low_eff,low_eff,2*low_eff],
                                 [low_eff,low_eff,low_eff,low_eff,1.,low_eff,2*low_eff]])

agent = ACER_agent(7, 5,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                      dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = True,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    buffer_limit         = 500 if not sl_rl else int(6e3) , 
                    rollout_len          = 2,
                    batch_size           = 128,     
                    c                    = 1.0, 
                    max_sentence_length  = 100,
                    steps_before_train   = 128 + 1)
agent.model = agent.model.to(device)
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
criterion = nn.CrossEntropyLoss().to(device)

# def evaluate(model, iterator, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
    
#     model.eval()
#     with torch.no_grad():
    
#         for batch in iterator:
#             predictions = model(batch.text)
            
#             loss = criterion(predictions, batch.label)
            
#             acc = categorical_accuracy(predictions, batch.label)

#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
        
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)

# %% 

n_trials = 2
epochs = 20
rewards = np.zeros((n_trials, len(train_iterator)*batch_size * epochs))
acc = np.zeros((n_trials, epochs))

best_valid_loss = float('inf')
loop = tqdm(range(n_trials))
for trial in loop:

    agent.reset()
    agent.model.NLP.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    agent.model.NLP.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
    agent.model.train()
    
    i = 0
    for e in range(epochs):
        
        train_loss_SL = 0
        train_acc_SL = 0        
        
        for batch in train_iterator:
            
            agent.optimizer_SL.zero_grad()
            predictions = agent.model.NLP(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)
            loss.backward(retain_graph= not sl_rl)
            agent.optimizer_SL.step()
            
            train_loss_SL += loss.item()
            train_acc_SL += acc.item()
            
            batch.label = batch.label.cpu().numpy().tolist()
            
            for state, dung in zip(predictions,batch.label):
    
                # Action selection
                action = agent.act(state)
                
                # Action perform
                dead = random.random() > weapon_in_dung_score[dung,action]
                reward = reward_die if dead else reward_win
                
                # Agent update and train
                with torch.no_grad(): prob = agent.model.RL.pi(state).cpu().numpy()
                
                # If SL and RL separated
                if sl_rl: state = state.detach().cpu().numpy()                
                
                agent.update(i, state, prob, action, reward); i+= 1
        
                # Storing the reward
                rewards[trial, i] = reward
            
        
        train_loss_SL /= len(train_iterator) 
        train_acc_SL  /= len(train_iterator)    
        
        acc[trial, e] = train_acc_SL
            
        loop.set_description(f'Epoch: {e} | Train Loss: {train_loss_SL:.3f} | Train Acc: {train_acc_SL*100:.2f}%'+\
                               f'| Mean recent rewards: {rewards[trial,:-100].mean():.3f}')
    
# %%
import matplotlib.pyplot as plt

cut = 20

col = "green"

m = smooth(rewards.mean(0))[cut:]
s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
line = plt.plot(m, alpha=0.7, label=agent.name,
                  color=col, lw=3)[0]
plt.fill_between(range(len(m)), m + s, m - s,
                    color=line.get_color(), alpha=0.2)
plt.hlines(reward_win, reward_win, len(rewards[0]), color = "chocolate", linestyles="--")
plt.hlines(reward_die, reward_die, len(rewards[0]), color = "chocolate", linestyles="--")
plt.ylim(reward_die-0.5, reward_win + 0.5)






# %% 

# class NLP_ActorCritic(nn.Module):

#     def __init__(self, k, action_dim, NLP):
#         nn.Module.__init__(self)   
#         # self.NLP = NLP_NN_EASY(vocab_dim, k)
#         self.NLP = NLP
#         self.RL  = ActorCritic(k, action_dim)
    
#     def pi(self, x, softmax_dim = 0): 
#         if x.dim() != 2: 
#             x = x.squeeze()
#         return self.RL.pi(self.NLP(x))
#     def q(self, x):                   
#         if x.dim() != 2: 
#             x = x.squeeze()
#         return self.RL.q(self.NLP(x))

# # Hyperparameters
# n_mission_per_episode   = 10    # Every episode is made of consecutive missions
# n_equip_can_take        = 1     # Equipement the explores has for every mission
# n_trials                = 2     # Trials for estimating performance (training) 
# n_test_trials           = 1000   # Trials for estimating performance (testing)   
# buffer_size             = int(5e3)  # Buffer size for memory cells of the algorithms
# batch_size              = 64
# episode_before_train    = batch_size + 1
# episode_count           = int(1e3)  # Number of episodes for training
# NNLP_env= env           = gym.make('nlp2020:nnlpDungeon-v0')
# NLP_env                 = gym.make('nlp2020:nlpDungeon-v0')

# # Setting equip
# NNLP_env.set_num_equip(n_equip_can_take)
# NLP_env.set_num_equip(n_equip_can_take)
# env.set_num_equip(n_equip_can_take)

# algs = {}
# # Create the data structure that contains all the stuff for train and test
# """
# {agent : (environment, array for storing rewards, train function,
#           test_function, color for plots, number of episode to run)}
# """

# # ACER NLP FULLY INFORMED
# agent = ACER_agent(env.observation_space.n, env.action_space.n,
#                 fully_informed       = True,
#                 nlp                  = True,
#                 learning_rate        = 0.0002,
#                 gamma                = 0.98,
#                 buffer_limit         = buffer_size , 
#                 rollout_len          = 3,
#                 batch_size           = batch_size,     
#                 c                    = 1.0, 
#                 max_sentence_length  = 100,
#                 episode_before_train = batch_size + 1)

# agent.device = device
# agent.model = NLP_ActorCritic(5, 7, model).to(device)
# agent.tokenize = lambda s : TEXT.process([TEXT.tokenize(s)], device = device).T

# algs[agent.name] = (agent, NLP_env, np.zeros((n_trials,episode_count)),
#                 train1, test1, "lime", episode_count) 

# # RANDOM AGENT
# algs["RandomAgent"] = (RandomAgent(env.action_space.n), NNLP_env, np.zeros((n_trials,episode_count)),
#           train1, test1, "red", episode_count) 


# # Running the experiment
# save = False;  load = False; load_reward = False;
# for _,(agent,env,rewards,train_func,_,_,episode_count) in algs.items():
#     loop = tqdm(range(n_trials))
#     for trial in loop:

#         if agent.name == "RandomAgent":
#               agent.reset() # Agent reset learning before starting another trial
#         else: agent.model = NLP_ActorCritic(5, 7, model)
#         if load: agent.load_model()
        
#         # Training loop for a certain number of episodes
#         train_func(agent, env, loop, episode_count, rewards, trial)
    
#     if save and agent.name != "RandomAgent": agent.save_model(algs[agent.name][2]) 

#     if load_reward:
#         old = np.loadtxt("./logs_nlp2020/"+agent.name+".txt")
#         if len(old.shape) == 1: old = old.reshape(1,-1)
#         new = algs[agent.name][2]
#         algs[agent.name][2] = np.hstack((old,new))


# # TRAINING PERFORMANCE
# for _,(agent,env,rewards,_,_,col,_) in algs.items():
#     cut = 20
#     m = smooth(rewards.mean(0))[cut:]
#     s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
#     line = plt.plot(m, alpha=0.7, label=agent.name,
#                       color=col, lw=3)[0]
#     plt.fill_between(range(len(m)), m + s, m - s,
#                         color=line.get_color(), alpha=0.2)
# plt.hlines(0, 0, episode_count, color = "chocolate", linestyles="--")
# plt.hlines(-n_mission_per_episode, 0, episode_count, color = "chocolate", linestyles="--")
# plt.ylim(-n_mission_per_episode-0.5, 0.5)
# plt.legend(); plt.show()


# from nlp2020.utils import multi_bar_plot
# # TESTING PERFORMANCE
# test_trials = {}
# for _,(agent, env,_,_,test_func,_,_) in algs.items():
#     agent.model.eval()
#     test_trials[agent.name] = np.zeros(n_test_trials, dtype = int)
#     loop = tqdm(range(n_test_trials), desc = f"{agent.name}"); loop.refresh()  
#     for trial in loop: test_func(agent, env, trial, test_trials)
# multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials)

















