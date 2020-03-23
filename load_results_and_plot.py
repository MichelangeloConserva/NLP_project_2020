# %% Loading
import matplotlib.pyplot as plt    
import pickle
import numpy as np

from collections import Counter

from nlp2020.utils import smooth

# directory = "./logs_nlp2020/notasexpected/"
directory = "./logs_nlp2020/"


with open(directory+"trials"+".pickle", "rb") as f:      test_trials = pickle.load(f)
with open(directory+"rewards_acc"+".pickle", "rb") as f: rr_dict = pickle.load(f)

# with open("./logs_nlp2020/trials.pickle", "wb") as f: pickle.dump(test_trials, f)
# with open("./logs_nlp2020/rewards_acc.pickle", "wb") as f: pickle.dump(rr_dict, f)

# Converting tuple to list
rr_dict = {k:list(v) for k,v in rr_dict.items()}

pairs_colors = [("gold","khaki"),
                # ("sienna","chocolate"),
                ("darkgreen","olive"),
                # ("olive","olivedrab"),
                ("brown","rosybrown"),
                ("navy","deepskyblue"),
                ("darkviolet","magenta")] 
algs_names = list(filter( lambda x : not "drop" in x and "Random" not in x, rr_dict.keys()))


styles = ['-', '--', '-.', ':', ':']

for i,name in enumerate(algs_names):
    rr_dict[name][-1], rr_dict[name+"_dropout"][-1] = pairs_colors[::-1][i]
    rr_dict[name].append(styles[i])
    rr_dict[name+"_dropout"].append(styles[i])
    
rr_dict["Random_NotInformed_NNLP"][-1] = "grey"
rr_dict["Random_NotInformed_NNLP"].append(styles[-1])


actual_labels = { 'ACERAgent_NotInformed_NLP_only_RL_dropout'    : "ACER_NLP_JustRL_dropout",
                  'ACERAgent_NotInformed_NLP_only_RL'            : "ACER_NLP_JustRL",
                  'ACERAgent_FullyInformed_NLP_SL_both_RL_dropout' : "ACER_NLP_SLandRL_dropout",
                  'ACERAgent_FullyInformed_NLP_SL_both_RL'         : "ACER_NLP_SLandRL",
                  'ACERAgent_FullyInformed_NLP_SL_sep_RL_dropout'  : "ACER_NLP_SL//RL_dropout",
                  'ACERAgent_FullyInformed_NLP_SL_sep_RL'          : "ACER_NLP_SL//RL",
                  'ACERAgent_NotInformed_NNLP_dropout'             : "ACER_NoContext_dropout",
                  'ACERAgent_NotInformed_NNLP'                     : "ACER_NoContext",
                  'Random_NotInformed_NNLP'                        : "Random", 
                  'Random'                                         : "Random", 
                  'ACERAgent_FullyInformed_NNLP_dropout'           : "ACER_DeterministicContext_dropout",
                  'ACERAgent_FullyInformed_NNLP'                   : "ACER_DeterministicContext",
                  }

reward_win = 10
reward_die = -10

# %% Plot performance in training all together
import seaborn as sns
sns.set(font_scale=0.8)

# TRAINING PERFORMANCE
cut = 20
wind = 320
my_dpi = 96
plt.figure(figsize=(1000/my_dpi, 500/my_dpi), dpi=my_dpi)
for ag in filter(lambda x: not "dropout" in x and not "Random" in x, test_trials.keys()):
    if "only" in ag: continue

    for agent_name in [ag, ag+"_dropout"]:
        rewards,acc_hist,col,style = rr_dict[agent_name]
        if len(rewards) == 0: continue
        rewards = np.array(rewards) / reward_win
        
        r_mean =  \
        np.vstack([r_trial.reshape(-1,wind).mean(1).tolist() for r_trial in rewards ])
        
        
        m = smooth(r_mean.mean(0), 50, r_mean.mean(0)[0])[cut:]
        s = (np.std(smooth(r_mean.T).T, axis=0)/np.sqrt(len(r_mean)))[cut:]
        line = plt.plot(m, alpha=1, label=actual_labels[agent_name],
                          color=col, lw=5, linestyle = style)[0]
        plt.fill_between(range(len(m)), m + s, m - s,
                            color=line.get_color(), alpha=0.2)
agent_name = "Random_NotInformed_NNLP"
rewards,acc_hist,col,style = rr_dict[agent_name]
rewards = np.array(rewards) / reward_win
r_mean =  \
np.vstack([r_trial.reshape(-1,wind).mean(1).tolist() for r_trial in rewards ])
m = smooth(r_mean.mean(0), 100, r_mean.mean(0)[0])[cut:]
s = (np.std(smooth(r_mean.T).T, axis=0)/np.sqrt(len(r_mean)))[cut:]
line = plt.plot(range(0,len(m),90), m[::90], alpha=1, label=actual_labels[agent_name],
                  color=col, lw=3, marker = "x", markersize = 10)[0]
plt.fill_between(range(len(m)), m + s, m - s,
                    color=line.get_color(), alpha=0.2)
plt.hlines(1, 1, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.hlines(-1, -1, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.ylim(-1-0.2, 1 + 0.1)
plt.xlim(0, plt.xticks()[0][-2])
plt.xticks(np.linspace(0, 3000, 6),[0,100,200,300,400,500])
plt.xlabel("Epoch")
plt.ylabel("Normalized Reward")
plt.legend(loc=0, ncol = 3); plt.show()


# %% Plot performance in training separated
import seaborn as sns
sns.set(font_scale=1.1)

main_list = ["ACERAgent_NotInformed_NLP_only_RL",
             "ACERAgent_FullyInformed_NLP_SL_sep_RL",
             "ACERAgent_NotInformed_NNLP",
             "Random_NotInformed_NNLP",
             "ACERAgent_FullyInformed_NNLP"]


cur_dict = {k:v for k,v in rr_dict.items() if k in main_list}
# TRAINING PERFORMANCE
cut = 20
wind = 320
my_dpi = 96
plt.figure(figsize=(1000/my_dpi, 500/my_dpi), dpi=my_dpi)
for agent_name,(rewards,acc_hist,col,style) in cur_dict.items():
    if len(rewards) == 0: continue
    rewards = np.array(rewards) / reward_win

    # to reduce the length of the time series we take the mean value every 100
    r_mean =  \
    np.vstack([r_trial.reshape(-1,wind).mean(1).tolist() for r_trial in rewards ])
    
    m = smooth(r_mean.mean(0), 50, r_mean.mean(0)[0])[cut:]
    s = (np.std(smooth(r_mean.T).T, axis=0)/np.sqrt(len(r_mean)))[cut:]
    line = plt.plot(m, alpha=1, label=actual_labels[agent_name],
                      color=col, lw=5, linestyle = style)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)
 
plt.hlines(1, 1, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.hlines(-1, -1, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.ylim(-1-0.2, 1 + 0.1)
plt.xlim(0, plt.xticks()[0][-2])
plt.xticks(np.linspace(0, 3000, 6),[0,100,200,300,400,500])
plt.xlabel("Epoch")
plt.ylabel("Normalized Reward")
plt.legend(loc=0, ncol = 3); plt.show()



# %% Accuracy during training
sns.set(font_scale=0.8)
my_dpi = 96
plt.figure(figsize=(500/my_dpi, 250/my_dpi), dpi=my_dpi)
for ag in filter(lambda x: not "dropout" in x and not "Random" in x, test_trials.keys()):
    for agent_name in [ag, ag+"_dropout"]:
        rewards,acc_hist,col,style = rr_dict[agent_name]
        
        if acc_hist.sum() == 0: continue
        
        cut = 20
        wind = 100
        
        m = smooth(acc_hist.mean(0), 100, acc_hist.mean(0)[0])[cut:][:50]
        s = (np.std(smooth(acc_hist.T).T, axis=0)/np.sqrt(len(acc_hist)))[cut:][:50]
        line = plt.plot(m, alpha=0.7, label=actual_labels[agent_name],
                          color=col, lw=3, linestyle = style)[0]
        plt.fill_between(range(len(m)), m + s, m - s,
                            color=line.get_color(), alpha=0.2)
     
        plt.hlines(1, 1, len(m), color = "chocolate", linestyles="--")
        plt.hlines(-1, -1, len(m), color = "chocolate", linestyles="--")
        plt.ylim(0.8, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc=0)
plt.xticks(plt.xticks()[0],(plt.xticks()[0]*10).astype(int))
plt.xlim(0,plt.xticks()[0][-2])
plt.show()

# %% Plot performance in testing

spacing = np.linspace(-0.6,0.6, len(test_trials))
width = spacing[1] - spacing[0]
missions = np.array([0,2])
ii = 0
sns.set(font_scale=0.85)
my_dpi = 96
plt.figure(figsize=(1000/my_dpi, 500/my_dpi), dpi=my_dpi)
for ag in filter(lambda x: not "dropout" in x and not "Random" in x, test_trials.keys()):
    
    for agent_name in [ag, ag+"_dropout"]:
        
        c = Counter(test_trials[agent_name])
        if reward_die not in c.keys(): c[reward_die] = 0
        
        c_sum = sum([v for v in c.values()])
        
        for k,v in c.items(): c[k] = v/c_sum
    
        assert round(sum(c.values()),5) == 1, round(sum(c.values()),5)
        
        col = rr_dict["Random_NotInformed_NNLP" if "Random" in agent_name else agent_name][-2]
        plt.bar(missions + spacing[ii], 
                [c[k] for k in sorted(c.keys())], width, label = actual_labels[agent_name], color = col, edgecolor="black")
        ii += 1
agent_name = "Random"
c = Counter(test_trials[agent_name])
if reward_die not in c.keys(): c[reward_die] = 0

c_sum = sum([v for v in c.values()])

for k,v in c.items(): c[k] = v/c_sum

assert round(sum(c.values()),5) == 1, round(sum(c.values()),5)

col = rr_dict["Random_NotInformed_NNLP" if "Random" in agent_name else agent_name][-2]
plt.bar(missions + spacing[ii], 
        [c[k] for k in sorted(c.keys())], width, label = actual_labels[agent_name], color = col, edgecolor="black")
 
plt.xlabel("Rewards")
plt.xticks(missions,[-1,1])
plt.legend(ncol=3)    
plt.ylim(0,1.3)
plt.show()




# %% 

























# %%    During train performance 
from os import listdir
from os.path import isfile, join
directory = "./logs_nlp2020/"


test_trials = {}

onlyfiles = [f for f in listdir(directory) if (isfile(join(directory, f)) and "trial" in f)]
for ff in onlyfiles:
    with open(directory+ff, "rb") as f: rs = pickle.load(f)
    agent_name = "_".join(ff.split("_")[1:-2])
    test_trials[agent_name] = rs


spacing = np.linspace(-1,1, len(test_trials)+1)
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
    
    col = test_trials["Random_NotInformed_NNLP" if "Random" in agent_name else agent_name][-1]
    plt.bar(missions + spacing[ii], 
            [c[k] for k in sorted(c.keys())], width, label = agent_name, color = col, edgecolor="black")
    ii += 1
plt.xlabel("Rewards")
plt.xticks(missions,[-1,1])
plt.legend()    
plt.show()




















