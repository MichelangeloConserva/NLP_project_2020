# %%
import  torch
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm
from collections import Counter

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.train_test_functions import train_f
from nlp2020.utils import  smooth, create_iterator


# Environment parameters
low_eff = 0.1
weapon_in_dung_score = np.array([[1.,low_eff,low_eff,low_eff,low_eff,low_eff,2*low_eff],
                                 [low_eff,1.,low_eff,low_eff,low_eff,low_eff,2*low_eff],
                                 [low_eff,low_eff,1.,low_eff,low_eff,low_eff,2*low_eff],
                                 [low_eff,low_eff,low_eff,1.,low_eff,low_eff,2*low_eff],
                                 [low_eff,low_eff,low_eff,low_eff,1.,low_eff,2*low_eff]])
reward_win = 1
reward_die = -1

# Training parameters
n_trials = 5
epochs = 20
sl_rl = True
batch_size = 1400
train_iterator, test_iterator, _, LABEL, TEXT = create_iterator("cuda", batch_size, int(2e3))

# NLP parameters
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
vocab_size     = len(TEXT.vocab)
embedding_dim  = 128
n_filters      = 500
filter_sizes   = [2,3,4]
dropout        = 0.1
pad_idx        = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]


# Create the data structure that contains all the stuff for train and test
algs = {}
"""
{name : (agent,environment, array for storing rewards, array for storing accuracy,
         train function, color for plots, number of episode to run)}
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
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "navy", epochs]

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
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "green", 
                    epochs]

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
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "skyblue", epochs]


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
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "cyan", epochs]

algs["Random"] = [RandomAgent(7), [], 
                  np.zeros((n_trials, epochs)), train_f, "red", epochs]

# Running the experiment
save = True;  load = False; load_reward = False;
for _,(agent,rewards,acc_hist,train_func,col,epochs) in algs.items():
    
    # if "Random" not in agent.name: print(agent.model)
    
    loop = tqdm(range(n_trials))
    for trial in loop:
        agent.loop = loop
        
        agent.reset() # Agent reset learning before starting another trial
        if load: 
            try:    agent.load_model()
            except: pass
      
        # Training loop for a certain number of episodes
        rewards.append(train_func(agent, loop, n_trials, epochs, train_iterator, acc_hist, rewards, trial))

    rewards = [rrr for rrr in rewards if rrr is not None]
    assert np.array(rewards).shape[0] == n_trials


# %%
# import seaborn as sns
# sns.set(font_scale=1.5)

# TRAINING PERFORMANCE
plt.figure()
for _,(agent,rewards,acc_hist,_,col,_) in algs.items():
    assert np.array(rewards).shape[0] == n_trials
    rewards = np.array(rewards)

    # np.save("./logs_nlp2020/"+agent.name.replace("/","__"), rewards)
    
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
for _,(agent,rewards,acc_hist,_,col,_) in algs.items():
    if "Random" not in agent.name:  agent.model = agent.model.eval()
    
    loop = tqdm(range(n_test_trials), desc = f"{agent.name}"); loop.refresh()  
    rs = []
    for trial in loop:
        for batch in test_iterator:
            rs += agent.act_and_train(batch, test = True)
    
    if "Random" not in agent.name:  
        test_trials[agent.name] = rs
        agent.model = agent.model.train()
    else:
        test_trials["Random"] = rs
    
# import pickle
# with open("./logs_nlp2020/trials", "wb") as f: pickle.dump(test_trials, f)

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
