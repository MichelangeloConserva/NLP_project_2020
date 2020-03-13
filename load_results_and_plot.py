# %% Loading
import matplotlib.pyplot as plt    
import pickle
import numpy as np

from collections import Counter

from nlp2020.utils import smooth

directory = "./logs_nlp2020/first_good/"

with open(directory+"trials_last.pickle", "rb") as f:      test_trials = pickle.load(f)
with open(directory+"rewards_acc_last.pickle", "rb") as f: rr_dict = pickle.load(f)

with open("./logs_nlp2020/trials.pickle", "wb") as f: pickle.dump(test_trials, f)
with open("./logs_nlp2020/rewards_acc.pickle", "wb") as f: pickle.dump(rr_dict, f)


# Converting tuple to list
rr_dict = {k:list(v) for k,v in rr_dict.items()}

pairs_colors = [("gold","khaki"),
                ("sienna","chocolate"),
                ("brown","rosybrown"),
                ("olive","olivedrab"),
                ("darkgreen","green"),
                ("navy","deepskyblue"),
                ("darkviolet","magenta")] 
algs_names = list(filter( lambda x : not "drop" in x and "Random" not in x, rr_dict.keys()))

for i,name in enumerate(algs_names):
    rr_dict[name][-1], rr_dict[name+"_dropout"][-1] = pairs_colors[::-1][i]
rr_dict["Random_NotInformed_NNLP"][-1] = "silver"

reward_win = 1
reward_die = -1

# %% Plot performance in training
# import seaborn as sns
# sns.set(font_scale=1.5)

# TRAINING PERFORMANCE
plt.figure()
for agent_name,(rewards,acc_hist,col) in rr_dict.items():
    
    rewards = np.array(rewards)

    cut = 20
    wind = 100
    
    # to reduce the length of the time series we take the mean value every 100
    r_mean =  \
    np.vstack([r_trial.reshape(-1,wind).mean(1).tolist() for r_trial in rewards ])
    
    m = smooth(r_mean.mean(0), 100, r_mean.mean(0)[0])[cut:]
    s = (np.std(smooth(r_mean.T).T, axis=0)/np.sqrt(len(r_mean)))[cut:]
    line = plt.plot(m, alpha=0.7, label=agent_name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)

 
plt.hlines(reward_win, reward_win, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.hlines(reward_die, reward_die, len(r_mean[0]), color = "chocolate", linestyles="--")
plt.ylim(reward_die-0.5, reward_win + 0.5)
plt.legend(loc=0); plt.show()


# %% Accuracy during training

plt.figure()
for agent_name,(rewards,acc_hist,col) in rr_dict.items():
    
    if acc_hist.sum() == 0: continue
    
    cut = 20
    wind = 100
    
    m = smooth(acc_hist.mean(0), 100, acc_hist.mean(0)[0])[cut:][:50]
    s = (np.std(smooth(acc_hist.T).T, axis=0)/np.sqrt(len(acc_hist)))[cut:][:50]
    line = plt.plot(m, alpha=0.7, label=agent_name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)
 
plt.hlines(reward_win, reward_win, len(m), color = "chocolate", linestyles="--")
plt.hlines(reward_die, reward_die, len(m), color = "chocolate", linestyles="--")
plt.ylim(0.8, 1)
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
    
    col = rr_dict["Random_NotInformed_NNLP" if "Random" in agent_name else agent_name][-1]
    plt.bar(missions + spacing[ii], 
            [c[k] for k in sorted(c.keys())], width, label = agent_name, color = col, edgecolor="black")
    ii += 1
plt.xlabel("Rewards")
plt.xticks(missions,[-1,1])
plt.legend()    
plt.show()


# %%    During train performance 
from os import listdir
from os.path import isfile, join
directory = "./logs_nlp2020/"


test_trials = {}

onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
for ff in onlyfiles:
    with open(directory+ff, "rb") as f: rs = pickle.load(f)
    agent_name = "_".join(ff.split("_")[1:-2])
    test_trials[agent_name] = rs


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
    
    col = rr_dict["Random_NotInformed_NNLP" if "Random" in agent_name else agent_name][-1]
    plt.bar(missions + spacing[ii], 
            [c[k] for k in sorted(c.keys())], width, label = agent_name, color = col, edgecolor="black")
    ii += 1
plt.xlabel("Rewards")
plt.xticks(missions,[-1,1])
plt.legend()    
plt.show()




















