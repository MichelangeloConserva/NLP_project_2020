import gym, torch
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)
from tqdm import tqdm

from nlp2020.agents.random_agent  import RandomAgent
from nlp2020.agents.dqn_agent     import DQN_agent
from nlp2020.agents.acer_agent    import ACER_agent
from nlp2020.agents.neural_linear_agent    import NLB
from nlp2020.utils                import smooth, multi_bar_plot
from nlp2020.train_test_functions import train1, test1
from nlp2020.utils                import get_vocab

# Hyperparameters
n_mission_per_episode   = 10    # Every episode is made of consecutive missions
n_equip_can_take        = 1     # Equipement the explores has for every mission
n_trials                = 2     # Trials for estimating performance (training) 
n_test_trials           = 500   # Trials for estimating performance (testing)   
buffer_size             = int(5e3)  # Buffer size for memory cells of the algorithms
batch_size              = 128
steps_before_train       = batch_size + 1
episode_count           = int(40e3)  # Number of episodes for training
# training_time           = 5 * 60 
env                     = gym.make('nlp2020:cbe-v0')
TEXT = get_vocab()
vocab_size     = len(TEXT.vocab)
embedding_dim  = 128
n_filters      = 300
filter_sizes   = [2,3,4]
dropout        = 0.1
pad_idx        = TEXT.vocab.stoi[TEXT.pad_token]

algs = {}
# Create the data structure that contains all the stuff for train and test
"""
{name : (agent,environment, array for storing rewards, train function,
          test_function, color for plots, number of episode to run)}
"""

# NEURAL LINEAR BANDIT
# agent = NLB(env.num_dung.n, env.action_space.n,
#                     vocab_size, embedding_dim, n_filters, filter_sizes,  
#                   dropout, pad_idx, TEXT,
#                   nlp = True)
# algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
#                 train1, test1, "pink", episode_count]


# ACER NLP FULLY INFORMED
agent = ACER_agent(env.num_dung.n, env.action_space.n,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                  dropout, pad_idx,TEXT,
                fully_informed       = True,
                nlp                  = True,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size , 
                rollout_len          = 2,
                batch_size           = batch_size,     
                c                    = 1.0, 
                max_sentence_length  = 100,
                steps_before_train = steps_before_train)
algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
                train1, test1, "lawngreen", episode_count]

# ACER NLP FULLY INFORMED
agent = ACER_agent(env.num_dung.n, env.action_space.n,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                  dropout, pad_idx,TEXT,
                fully_informed       = False,
                nlp                  = True,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size , 
                rollout_len          = 2,
                batch_size           = batch_size,     
                c                    = 1.0, 
                max_sentence_length  = 100,
                steps_before_train = steps_before_train)
algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
                train1, test1, "olive", episode_count]

# ACER NOT NLP FULLY INFORMED
agent = ACER_agent(env.num_dung.n, env.action_space.n,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                  dropout, pad_idx,TEXT,
                fully_informed       = True,
                nlp                  = False,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size, 
                rollout_len          = 2,
                batch_size           = batch_size,
                c                    = 1.0, 
                max_sentence_length  = 100,
                steps_before_train = steps_before_train
                )
algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
                    train1, test1, "g", episode_count]
      
# ACER NOT FULLY INFORMED
agent = ACER_agent(env.num_dung.n, env.action_space.n,
                    vocab_size, embedding_dim, n_filters, filter_sizes,  
                  dropout, pad_idx,TEXT,
                fully_informed       = False,
                nlp                  = False,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size, 
                rollout_len          = 2,
                batch_size           = batch_size,
                c                    = 1.0,
                max_sentence_length  = 100,
                steps_before_train = steps_before_train         
                )
algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
                train1, test1, "darkgreen", episode_count]

# DQN NLP FULLY INFORMED
# agent = DQN_agent(env.num_dung.n, env.action_space.n,
#                     vocab_size, embedding_dim, n_filters, filter_sizes,  
#                   dropout, pad_idx,TEXT,nlp = True, fully_informed = False,
#                 batch_size = batch_size, gamma = 0.999, eps_end = 0.001,
#                 eps_decay = int(episode_count//1.11) , target_update = 100, buffer_size = buffer_size,
#                 max_sentence_length = 95  )              
# algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
#                 train1, test1, "aqua", episode_count]

# # DQN NLP FULLY INFORMED
# agent = DQN_agent(env.num_dung.n, env.action_space.n,
#                     vocab_size, embedding_dim, n_filters, filter_sizes,  
#                   dropout, pad_idx,TEXT,nlp = True,
#                 batch_size = batch_size, gamma = 0.999, eps_end = 0.001,
#                 eps_decay = int(episode_count//1.11) , target_update = 100, buffer_size = buffer_size,
#                 max_sentence_length = 95  )              
# algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
#                 train1, test1, "aqua", episode_count]

# # DQN NOT NLP FULLY INFORMED
# agent = DQN_agent(env.num_dung.n, env.action_space.n, 
#                     vocab_size, embedding_dim, n_filters, filter_sizes,  
#                   dropout, pad_idx,TEXT,nlp = False, 
#                 batch_size = batch_size, gamma = 0.999, eps_start = 0.9,
#                 eps_end = 0.01, eps_decay = int(episode_count//1.11), target_update = 100,
#                 buffer_size = buffer_size, max_sentence_length = 100 )
# algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
#                 train1, test1, "steelblue", episode_count]
    
# # DQN NOT FULLY INFORMED
# agent = DQN_agent(env.num_dung.n, env.action_space.n,
#                     vocab_size, embedding_dim, n_filters, filter_sizes,  
#                   dropout, pad_idx,TEXT, fully_informed = False,
#                 nlp = False, batch_size = batch_size, gamma = 0.999, eps_start = 0.9,
#                 eps_end = 0.01, eps_decay = int(episode_count//1.11), target_update = 100,
#                 buffer_size = buffer_size, max_sentence_length = 100)
# algs[agent.name] = [agent, np.zeros((n_trials,episode_count)),
#                 train1, test1, "navy", episode_count]

# RANDOM AGENT
algs["RandomAgent"] = [RandomAgent(env.action_space.n), np.zeros((n_trials,episode_count)),
          train1, test1, "red", episode_count]


# Running the experiment
save = True;  load = False; load_reward = False;
for _,(agent,rewards,train_func,_,col,episode_count) in algs.items():
    
    loop = tqdm(range(n_trials))
    for trial in loop:
        agent.loop = loop
        
        agent.reset() # Agent reset learning before starting another trial
        if load: 
            try:    agent.load_model()
            except: pass
      
        # Training loop for a certain number of episodes
        train_func(agent, env, loop, episode_count, rewards, trial)
        
        
    if load_reward and agent.name != "RandomAgent":
        old = np.loadtxt("./logs_nlp2020/"+agent.name+".txt")
        if len(old.shape) == 1: old = old.reshape(1,-1)
        new = algs[agent.name][2]
        algs[agent.name][2] = np.hstack((old,new))
    
    if save and agent.name != "RandomAgent": agent.save_model(algs[agent.name][1])            
        
    

# algs["ACERAgent_FullyInformed_NLP"][-2] = "lawngreen"

import seaborn as sns
sns.set(font_scale=1.5)

# TRAINING PERFORMANCE
for _,(agent,rewards,_,_,col,_) in algs.items():
    if "DQN" in agent.name: continue
    
    cut = 20
    m = smooth(rewards.mean(0))[cut:]
    s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)
plt.hlines(env.reward_win, env.reward_win, episode_count, color = "chocolate", linestyles="--")
plt.hlines(env.reward_die, env.reward_die, episode_count, color = "chocolate", linestyles="--")
plt.ylim(env.reward_die-0.5, env.reward_win + 0.5)
plt.legend(loc=3); plt.show()


# TESTING PERFORMANCE
test_trials = {}
for _,(agent,_,_,test_func,_,_) in algs.items():
    if "DQN" in agent.name: continue
    
    if "Random" not in agent.name:  agent.model = agent.model.eval()
    test_trials[agent.name] = np.zeros(n_test_trials, dtype = int)
    loop = tqdm(range(n_test_trials), desc = f"{agent.name}"); loop.refresh()  
    for trial in loop: test_func(agent, env, trial, test_trials)
    if "Random" not in agent.name:  agent.model = agent.model.train()
    
    
# Multi bars plot
from collections import Counter

spacing = np.linspace(-1,1, 5)
width = spacing[1] - spacing[0]
missions = np.arange((2)*4, step = 4)
ii = 0
for (i,(_,(agent,_,_,_,col,_))) in enumerate(algs.items()):
    if "DQN" in agent.name: continue

    c = Counter(test_trials[agent.name])
    for k,v in c.items(): c[k] = v/n_test_trials

    assert round(sum(c.values()),5) == 1
    
    plt.bar(missions + spacing[ii], 
            list(c.values()), width, label = agent.name, color = col, edgecolor="black")
    ii += 1
plt.xlabel("Consecutive mission, i.e. length of the episode")
plt.xticks(missions,[-1,1])
plt.legend()    
plt.show()
    
plt.figure()
multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials)




if False:

    
    import pickle 
    
    
    filehandler = open("algs", 'w') 
    pickle.dump(algs, filehandler)
        
        
    agent = algs["ACERAgent_FullyInformed_NLP"][0]
    agent.tokenize = None
    
    filehandler = open("ACERAgent_FullyInformed_NLP", 'w') 
    pickle.dump(agent, filehandler)



    
# =============================================================================
# DEBUGGER
# =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env.reset()
    for _,(agent,_,_,_,_,_) in algs.items():
        if "Random" in agent.name: continue
        
        if agent.nlp:
            state = env.dungeon_description
            state = torch.tensor(agent.tokenize(state)).to(device)
        else:
            state = env.dung_descr
            # state = torch.tensor(state).to(device)


        print(agent)
        agent.act(state,test = True, printt = True)


agent = algs["ACERAgent_FullyInformed_NLP"][0]
env.reset()
ident = env.dung_descr
state = env.dungeon_description; state = torch.tensor(agent.tokenize(state)).to(device).view(1,-1)
print(ident,agent.model.NLP(state).detach().cpu().numpy().round(2))

agent = algs["ACERAgent_NotInformed_NLP"][0]
env.reset()
ident = env.dung_descr
state = env.dungeon_description; state = torch.tensor(agent.tokenize(state)).to(device).view(1,-1)
print(ident,agent.model.NLP(state).detach().cpu().numpy().round(2))









agent = algs["NLBAgent_FullyInformed_NLP"][0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env.reset()
ident = env.dung_descr
state = env.dungeon_description
state = torch.tensor(agent.tokenize(state)).to(device).view(1,-1)

print(ident,agent.bnn(state),agent.bnn(state).argmax())


agent.data_h.get_batch_with_weights(1)








