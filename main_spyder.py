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

torch.manual_seed(0)
np.random.seed(0)

#### STARTED AT 15:47 ######


# Environment parameters
low_eff = 0.01
med_eff = 0.3
weapon_in_dung_score = np.array([[1.,low_eff,low_eff,low_eff,low_eff,low_eff,med_eff],
                                 [low_eff,1.,low_eff,low_eff,low_eff,low_eff,med_eff],
                                 [low_eff,low_eff,1.,low_eff,low_eff,low_eff,med_eff],
                                 [low_eff,low_eff,low_eff,1.,low_eff,low_eff,med_eff],
                                 [low_eff,low_eff,low_eff,low_eff,1.,low_eff,med_eff]])
reward_win = 10
reward_die = -10

# Training parameters
n_trials = 5
epochs = 500
batch_size = 256
train_iterator, test_iterator, _, LABEL, TEXT = create_iterator("cuda", batch_size, int(2e3))

# NLP parameters
INPUT_DIM      = len(TEXT.vocab)
OUTPUT_DIM     = len(LABEL.vocab)
vocab_size     = len(TEXT.vocab)
embedding_dim  = 128
n_filters      = 350
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
# ACER NLP ONLY RL WITH DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    nlp                  = True,
                    sl_separated_rl      = False,
                    only_rl              = True,
                    fully_informed       = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0.25)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "firebrick", epochs]

# ACER NLP ONLY RL WITHOUT DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    nlp                  = True,
                    sl_separated_rl      = False,
                    only_rl              = True,
                    fully_informed       = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "red", epochs]

# ACER NLP SL AND SL+RL WITH DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    nlp                  = True,
                    sl_separated_rl      = False,
                    only_rl              = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0.25)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "navy", epochs]


# ACER NLP SL AND SL+RL WITHOUT DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    nlp                  = True,
                    sl_separated_rl      = False,
                    only_rl              = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "cyan", epochs]


# ACER NLP SL AND RL (SEPARATED) WITH DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    nlp                  = True,
                    sl_separated_rl      = True,
                    only_rl              = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0.25)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "green", epochs]

# ACER NLP SL AND RL (SEPARATED) WITHOUT DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    nlp                  = True,
                    sl_separated_rl      = True,
                    only_rl              = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "springgreen", 
                    epochs]

# ACER NOT INFORMED WITH DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    fully_informed       = False,
                    nlp                  = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0.25)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "springgreen", 
                    epochs]

# ACER NOT INFORMED WITHOUT DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    fully_informed       = False,
                    nlp                  = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "springgreen", 
                    epochs]

# ACER FULLY INFORMED WITH DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0.25)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "springgreen", 
                    epochs]

# ACER FULLY INFORMED WITH DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0.25)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "springgreen", 
                    epochs]

# ACER FULLY WITHOUT DROPOUT
agent = ACER_agent(5, 7, vocab_size, embedding_dim, n_filters, filter_sizes,  
                    dropout, pad_idx,TEXT,
                    fully_informed       = True,
                    nlp                  = False,
                    learning_rate        = 0.0002,
                    gamma                = 0.98,
                    c                    = 1.0,
                    dp_rl                = 0)
algs[agent.name] = [agent, [], np.zeros((n_trials, epochs)), train_f, "springgreen", 
                    epochs]

algs["Random"] = [RandomAgent(7), [], 
                  np.zeros((n_trials, epochs)), train_f, "red", epochs]


# %%

# for k in list(algs.keys()):
#     if not "drop" in k: algs.pop(k)

# Running the experiment
save = True;  load = False; load_reward = False;
for _,(agent,rewards,acc_hist,train_func,col,epochs) in algs.items():
    
    # try:
    #     if "dropout" not in agent.name:
    #         print(agent.name,"\n",agent.model)
    # except:
    #     pass
    
    
    if len(rewards) != 0: continue
    
    agent.store_env_vars(weapon_in_dung_score = weapon_in_dung_score,
                         reward_win = reward_win,
                         reward_die = reward_die)
    
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


# %% Additional training
            
            
epochs = 100
save = True;  load = False; load_reward = False;
for _,(agent,rewards,acc_hist,train_func,col,epochs) in algs.items():
    
    # if len(rewards) > 0: continue
    
    # try:
    #     if "dropout" not in agent.name:
    #         print(agent.name,"\n",agent.model)
    # except:
    #     pass
    
    # agent.store_env_vars(weapon_in_dung_score = weapon_in_dung_score,
    #                      reward_win = reward_win,
    #                      reward_die = reward_die)
    
    # if "Random" not in agent.name: print(agent.model)
    
    loop = tqdm(range(n_trials))
    for trial in loop:
        agent.loop = loop
        
        agent.reset() # Agent reset learning before starting another trial
        if load: 
            try:    agent.load_model()
            except: pass
      
        # Training loop for a certain number of episodes
        try:
            rewards[trial] += train_func(agent, loop, n_trials, epochs, 
                                         train_iterator, acc_hist, rewards, trial)
        except:
            loop.set_description("Gradient vanished")
            
# %% TESTING
    
n_test_trials = 10
test_trials = {}
for _,(agent,rewards,acc_hist,_,col,_) in algs.items():
    
    if len(rewards) == 0: continue
    
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
    

# %% Saving
import pickle

directory = "./logs_nlp2020/"
iii = ""

with open(directory+"trials"+iii+".pickle", "rb") as f:      test_trials_other = pickle.load(f)
with open(directory+"rewards_acc"+iii+".pickle", "rb") as f: rr_dict_other = pickle.load(f)

for k,v in test_trials_other.items():
    test_trials[k] = v



with open("./logs_nlp2020/trials.pickle", "wb") as f: pickle.dump(test_trials, f)
with open("./logs_nlp2020/rewards_acc.pickle", "wb") as f: 
    rr_dict = {}
    for _,(agent,rewards,acc_hist,_,col,_) in algs.items():
        rr_dict[agent.name] = [rewards, acc_hist, col]
        
    for k,v in rr_dict_other.items():
        if len(rewards) == 0: continue
        rr_dict[k] = v
        
    pickle.dump(rr_dict, f)






# %% Plot performance in training

# TRAINING PERFORMANCE
plt.figure()
for agent_name,(rewards,acc_hist,col) in rr_dict.items():
    
    rewards = np.array(rewards) / reward_win
    # assert rewards.shape[0] == n_trials
    # if len(rewards) == 0: continue

    cut = 20
    wind = 100
    
    r_mean = \
    np.vstack([r_trial.reshape(-1,wind).mean(1).tolist() for r_trial in rewards ])
    
    m = smooth(r_mean.mean(0), 100, r_mean.mean(0)[0])[cut:]
    s = (np.std(smooth(r_mean.T).T, axis=0)/np.sqrt(len(r_mean)))[cut:]
    line = plt.plot(m, alpha=0.7, label=agent_name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)

 
plt.hlines(1, 1, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.hlines(-1, -1, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.ylim(-1-0.5, 1 + 0.5)
plt.legend(loc=0); plt.show()


# %% Plot performance in testing

spacing = np.linspace(-1,1, len(test_trials))
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
plt.xlabel("Rewards")
plt.xticks(missions,[-1,1])
plt.legend()    
plt.show()






