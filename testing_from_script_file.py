! pip install --upgrade git+https://MichelangeloConserva:NLP_project_2020@github.com/MichelangeloConserva/NLP_project_2020.git

import gym
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.dungeon_creator import DungeonCreator
from nlp2020.utils import smooth

"""
Without the grid we define the episode as a maximum of 10 missions. 
If the agents dies then the episode ends.
"""

# Hyperparameters
n_mission_per_episode = 10 
n_equip_can_take = 2
n_trials = 20
episode_count = 1000
env = gym.make('nlp2020:nnlpDungeon-v0')

# Create environments, agents and storing array
algs = {}
algs[RandomAgent(env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
                                         np.zeros((n_trials,episode_count)),
                                         "red")
algs[DQN_agent(env.observation_space.n,
               env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
                                         np.zeros((n_trials,episode_count)),
                                         "blue")
# algs[ACER_agent(env.observation_space.n,
#                env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
#                                          np.zeros((n_trials,episode_count)),
#                                          "green")

# Running the experiment
loop = tqdm(range(n_trials))
for trial in loop:
    for agent,(env,rewards,_) in algs.items():
        
        # Agent reset learning before starting another trial
        agent.reset()
        
        for i in range(episode_count):
            
            # Start of the episode
            agent.start_episode()
            state = env.reset(); done = False; cum_reward = 0
            
            for t in range(n_mission_per_episode):
            
                    # Action selection
                    action = agent.act(state)
                    
                    # Action perform
                    next_state, reward, done, _ = env.step(action)
                    cum_reward += reward
        
                    # Observe new state
                    if not done: next_state = state
                    else: next_state = None   
                    
                    # Agent update and train
                    agent.update(i, state, action, next_state, reward)
    
                    # Move to the next state
                    state = next_state                            
                    if done: break
            
             # End of the episode
            rewards[trial, i] = cum_reward
            agent.end_episode()
            
                                       

for agent,(env,rewards,col) in algs.items():
    
    # m = smooth(rewards.mean(0))
    # s = np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards))
    # line = plt.plot(m, alpha=0.7, label=agent.name,
    #                   color=col, lw=3)[0]
    # plt.fill_between(range(len(m)), m + s, m - s,
    #                    color=line.get_color(), alpha=0.2)
    m = rewards.mean(1)
    s = rewards.std(1)
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s/2, m - s/2,
                       color=line.get_color(), alpha=0.2)
plt.legend()







             



fig,(ax1,ax2) = plt.subplots(1,2)
for agent,(env,rewards,col) in algs.items():
    
    # m = smooth(rewards.mean(0))
    # s = np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards))
    # line = ax1.plot(m, alpha=0.7, label=agent.name,
    #                   color=col, lw=3)[0]
    # ax1.fill_between(range(len(m)), m + s, m - s,
    #                    color=line.get_color(), alpha=0.2)
    
    m = rewards.mean(0)
    s = rewards.std(0)
    line = ax1.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    ax1.fill_between(range(len(m)), m + s, m - s,
                       color=line.get_color(), alpha=0.2)    
    
ax1.legend()
ax2.legend()
    







# =============================================================================
# AC3 agent (Fully informed)
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import gym
from nlp2020.agents.ac3_agent import Net, SharedAdam, v_wrap, Worker
import torch.multiprocessing as mp
from time import time

env = gym.make('nlp2020:nnlpDungeon-v0')



N_S = env.observation_space.n
N_A = env.action_space.n        
num_missions = 20
gnet = Net(N_S, N_A)        # global network
gnet.share_memory()         # share the global parameters in multiprocessing
opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

# parallel training
workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, 
                  env, num_missions, max_ep = 10000) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []                    # record episode reward to plot

start = time()
while True:
    r = res_queue.get()
    if r is None: break
    
    
[w.join() for w in workers]
 



with open("max_mission.txt", "r") as f: max_m = list(map(int, f.read().split(",")))
    

plt.subplot(1,2,1)
a,b = np.unique(max_m,return_counts=True); plt.bar(a,b/b.sum())
plt.xticks(range(num_missions),range(1,num_missions+1))
plt.title("Max epochs = 4000")

plt.subplot(1,2,2)
plt.plot(res); plt.ylabel('Moving average ep reward'); plt.xlabel('Step')





# =============================================================================
# AC3 agent (not informed)
# =============================================================================

from nlp2020.agents.ac3_agent import Net, SharedAdam, v_wrap, Worker
import torch.multiprocessing as mp
from time import time

env = gym.make('nlp2020:nnlpDungeon-v0')

N_S = env.observation_space.n
N_A = env.action_space.n        
num_missions = 20

gnet = Net(N_S, N_A)        # global network
gnet.share_memory()         # share the global parameters in multiprocessing
opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

# parallel training
workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, 
                  creator, num_missions, max_ep = 1000) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []                    # record episode reward to plot

start = time()
while True:
    r = res_queue.get()
    if r is not None: res.append(r)
    else: break
    if time() - start > 60: break
        
    
[w.join() for w in workers]

import matplotlib.pyplot as plt

with open("max_mission.txt", "r") as f:
    max_m = list(map(int, f.read().split(",")))


plt.subplot(1,2,1)
a,b = np.unique(max_m,return_counts=True)
plt.bar(a,b/b.sum())
plt.xticks(range(num_missions),range(1,num_missions+1))
plt.title("Max epochs = 4000")

plt.subplot(1,2,2)
plt.plot(res)
plt.ylabel('Moving average ep reward')
plt.xlabel('Step')
plt.show()














































# =============================================================================
# Sanity check using cartpole
# =============================================================================


import gym
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.dungeon_creator import DungeonCreator
from nlp2020.utils import smooth

"""
Without the grid we define the episode as a maximum of 10 missions. 
If the agents dies then the episode ends.
"""

# Hyperparameters
n_mission_per_episode = 10 
n_equip_can_take = 2
n_trials = 10
episode_count = 2000
env = gym.make('gym:CartPole-v0')

# Create environments, agents and storing array
algs = {}

algs[ACER_agent(4,
                env.action_space.n)] = (gym.make('gym:CartPole-v0'),
                                          np.zeros((n_trials,episode_count)),
                                          "green")
algs[RandomAgent(env.action_space.n)] = (gym.make('gym:CartPole-v0'),
                                         np.zeros((n_trials,episode_count)),
                                         "red")
algs[DQN_agent(4,
               env.action_space.n)] = (gym.make('gym:CartPole-v0'),
                                         np.zeros((n_trials,episode_count)),
                                         "blue")

# Running the experiment
loop = tqdm(range(n_trials))
for trial in loop:
    for agent,(env,rewards,_) in algs.items():
        loop.set_description(agent.name); loop.refresh()
        
        # Agent reset learning before starting another trial
        agent.reset()
        
        for i in range(episode_count):
            
            # Start of the episode
            agent.start_episode()
            state = env.reset(); done = False; cum_reward = 0
            
            
            while not done:
                    # Action selection
                    action = agent.act(state)
                    
                    # Action perform
                    next_state, reward, done, _ = env.step(action)
                    cum_reward += reward
        
                    # Observe new state
                    if not done: next_state = state
                    else: next_state = None   
                    
                    # Agent update and train
                    agent.update(i, state, action, next_state, reward)
    
                    # Move to the next state
                    state = next_state                            
            
             # End of the episode
            rewards[trial, i] = cum_reward
            agent.end_episode()
            
                                       

for agent,(env,rewards,col) in algs.items():
    
    # m = smooth(rewards.mean(0))
    # s = np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards))
    # line = plt.plot(m, alpha=0.7, label=agent.name,
    #                   color=col, lw=3)[0]
    # plt.fill_between(range(len(m)), m + s, m - s,
    #                    color=line.get_color(), alpha=0.2)
    m = rewards.mean(0)
    s = rewards.std(0)
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s/2, m - s/2,
                       color=line.get_color(), alpha=0.2)
plt.legend()

