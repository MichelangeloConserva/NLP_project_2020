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
from nlp2020.train_test_functions import train2, test1
from nlp2020.architectures import CNN, ActorCritic, NLP_ActorCritic
from nlp2020.utils import ACER_agent, create_dataset, Random_agent

reward_win = 1
reward_die = -1
sl_rl = True

SEED = 1234
batch_size = 1400
device = torch.device("cuda")
train_iterator, test_iterator, _, LABEL, TEXT = create_iterator(device, batch_size, int(2e3))

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
vocab_size     = len(TEXT.vocab)
embedding_dim  = 128
n_filters      = 500
filter_sizes   = [2,3,4]
dropout        = 0.1
pad_idx        = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

n_trials = 5
epochs = 200

algs = {}
# Create the data structure that contains all the stuff for train and test
"""
{name : (agent,environment, array for storing rewards, train function,
          test_function, color for plots, number of episode to run)}
"""

# ACER NLP FULLY INFORMED
agent = ACER_agent(5, 7,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                      dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = True,
                    sl_rl                = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    buffer_limit         = 500 if not sl_rl else int(6e3) , 
                    rollout_len          = 2,
                    batch_size           = 128,     
                    c                    = 1.0, 
                    max_sentence_length  = 100,
                    steps_before_train   = 128 + 1)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train2, test1, "navy", epochs]

# ACER NLP FULLY INFORMED
agent = ACER_agent(5, 7,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                      dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = False,
                    sl_rl                = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    buffer_limit         = 500 if not sl_rl else int(6e3) , 
                    rollout_len          = 2,
                    batch_size           = 128,     
                    c                    = 1.0, 
                    max_sentence_length  = 100,
                    steps_before_train   = 128 + 1)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train2, test1, "green", epochs]

# ACER NLP FULLY INFORMED
agent = ACER_agent(5, 7,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                      dropout, pad_idx,TEXT,
                    fully_informed       = False,
                    nlp                  = False,
                    sl_rl                = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    buffer_limit         = 500 if not sl_rl else int(6e3) , 
                    rollout_len          = 2,
                    batch_size           = 128,     
                    c                    = 1.0, 
                    max_sentence_length  = 100,
                    steps_before_train   = 128 + 1)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train2, test1, "skyblue", epochs]


# ACER NLP FULLY INFORMED
agent = ACER_agent(5, 7,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                      dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = True,
                    sl_rl                = True,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    buffer_limit         = 500 if not sl_rl else int(6e3) , 
                    rollout_len          = 2,
                    batch_size           = 128,     
                    c                    = 1.0, 
                    max_sentence_length  = 100,
                    steps_before_train   = 128 + 1)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train2, test1, "cyan", epochs]

algs["Random"] = [Random_agent(7), [], np.zeros((n_trials, epochs)), train2, test1, "red", epochs]


# Running the experiment
save = True;  load = False; load_reward = False;
for _,(agent,rewards,acc_hist,train_func,_,col,epochs) in algs.items():
    
    if "Random" not in agent.name: print(agent.model)
    
    loop = tqdm(range(n_trials))
    for trial in loop:
        agent.loop = loop
        
        agent.reset() # Agent reset learning before starting another trial
        if load: 
            try:    agent.load_model()
            except: pass
      
        # Training loop for a certain number of episodes
        rewards.append(train_func(agent, loop, n_trials, epochs, train_iterator, acc_hist, rewards, trial))


    assert np.array(rewards).shape[0] == n_trials

# %%
import seaborn as sns
sns.set(font_scale=1.5)

# TRAINING PERFORMANCE
plt.figure()
for _,(agent,rewards,acc_hist,_,_,col,_) in algs.items():
    rewards = np.array(rewards)

    np.save("./logs_nlp2020/"+agent.name.replace("/","__"), rewards)
    
    cut = 20
    m = smooth(rewards.mean(0))[cut:]
    s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)
    
plt.hlines(reward_win, reward_win, len(rewards[0]), color = "chocolate", linestyles="--")
plt.hlines(reward_die, reward_die, len(rewards[0]), color = "chocolate", linestyles="--")
plt.ylim(reward_die-0.5, reward_win + 0.5)
plt.legend(loc=0); plt.show()


n_test_trials = 10
test_trials = {}
for _,(agent,rewards,acc_hist,_,_,col,_) in algs.items():
    
    if "Random" not in agent.name:  agent.model = agent.model.eval()
    
    loop = tqdm(range(n_test_trials), desc = f"{agent.name}"); loop.refresh()  
    rs = []
    for trial in loop:
        for batch in test_iterator:
            r = agent.act_and_train(batch, test = True)
            rs += r
    
    
    if "Random" not in agent.name:  
        test_trials[agent.name] = rs
        agent.model = agent.model.train()
    else:
        test_trials["Random"] = rs
    
import pickle
with open("./logs_nlp2020/trials", "wb") as f: pickle.dump(test_trials, f)

    
from collections import Counter

spacing = np.linspace(-1,1, 5)
width = spacing[1] - spacing[0]
missions = np.arange((2)*4, step = 4)
ii = 0
plt.figure()
for agent_name in test_trials.keys():
    c = Counter(test_trials[agent_name])
    if reward_die not in c.keys(): c[reward_die] = 0
    
    c_sum = sum([v for v in c.values()])
    
    for k,v in c.items(): c[k] = v/c_sum

    assert round(sum(c.values()),5) == 1, round(sum(c.values()),5)
    
    col = algs[agent_name][-2]
    plt.bar(missions + spacing[ii], 
            [c[k] for k in sorted(c.keys())], width, label = agent_name, color = col, edgecolor="black")
    ii += 1
plt.xlabel("Consecutive mission, i.e. length of the episode")
plt.xticks(missions,[-1,1])
plt.legend()    
plt.show()



# best_valid_loss = float('inf')
# loop = tqdm(range(n_trials))
# for trial in loop:

#     agent.reset()
#     agent.model.NLP.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
#     agent.model.NLP.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)
    
#     agent.model.train()
    
#     i = 0
#     for e in range(epochs):
        
#         train_loss_SL = 0
#         train_acc_SL = 0        
        
#         for batch in train_iterator:
            
#             agent.optimizer_SL.zero_grad()
#             predictions = agent.model.NLP(batch.text)
#             loss = criterion(predictions, batch.label)
#             acc = categorical_accuracy(predictions, batch.label)
#             loss.backward(retain_graph= not sl_rl)
#             agent.optimizer_SL.step()
            
#             train_loss_SL += loss.item()
#             train_acc_SL += acc.item()
            
#             batch.label = batch.label.cpu().numpy().tolist()
            
#             for state, dung in zip(predictions,batch.label):
    
#                 # Action selection
#                 action = agent.act(state)
                
#                 # Action perform
#                 dead = random.random() > weapon_in_dung_score[dung,action]
#                 reward = reward_die if dead else reward_win
                
#                 # Agent update and train
#                 with torch.no_grad(): prob = agent.model.RL.pi(state).cpu().numpy()
                
#                 # If SL and RL separated
#                 if sl_rl: state = state.detach().cpu().numpy()                
                
#                 agent.update(i, state, prob, action, reward); i+= 1
        
#                 # Storing the reward
#                 rewards[trial, i] = reward
            
        
#         train_loss_SL /= len(train_iterator) 
#         train_acc_SL  /= len(train_iterator)    
        
#         acc_hist[trial, e] = train_acc_SL
            
#         loop.set_description(f'Epoch: {e} | Train Loss: {train_loss_SL:.3f} | Train Acc: {train_acc_SL*100:.2f}%'+\
#                                f'| Mean recent rewards: {rewards[trial,:-100].mean():.3f}')
    

