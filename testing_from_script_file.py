! pip install --upgrade git+https://MichelangeloConserva:NLP_project_2020@github.com/MichelangeloConserva/NLP_project_2020.git

# For selecting equipments we use multilabel classification style
# i.e. we put a sigmoid on the final layer and take highest k



import gym
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.dungeon_creator import DungeonCreator
from nlp2020.utils import smooth

"""
Without the grid we define the episode as a maximum of 10 missions. 
If the agents dies then the episode ends.
"""

# =============================================================================
# Random agent
# =============================================================================

# Hyperparameters
n_mission_per_episode = 10 
n_equip_can_take = 2
n_trials = 10
episode_count = 100
env = gym.make('nlp2020:nnlpDungeon-v0')

# Create environments, agents and storing array
algs = {}
algs[RandomAgent(env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
                                         np.zeros((n_trials,episode_count)),
                                         np.zeros((n_trials,episode_count)),
                                         "red")
algs[DQN_agent(env.observation_space.n,
               env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
                                         np.zeros((n_trials,episode_count)),
                                         np.zeros((n_trials,episode_count)),
                                         "blue")


for trial in range(n_trials):
    done = False
    reward = 0
    
    for i in range(episode_count):
        
        for agent,(env,rewards,rewards,_) in algs.items():
            
            state = env.reset()
            
            cum_reward = 0
            for t in range(n_mission_per_episode):
                
                
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                
                cum_reward += reward
                
                # Observe new state
                if not done: next_state = state
                else: next_state = None            
                
                agent.update(i, state, action, next_state, reward)
                
                # Move to the next state
                state = next_state            
                
                if done:
                    best_mission[trial, i] = t
                    rewards[trial, i] = cum_reward
                    break
    env.close()



for agent,(env,rewards,rewards,col) in algs.items():
    
    m = smooth(rewards.mean(0))
    s = np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards))
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                       color=line.get_color(), alpha=0.2)
plt.legend()
    
    





# =============================================================================
# AC3 agent (Fully informed)
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
                  env, num_missions, max_ep = 10000) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []                    # record episode reward to plot

start = time()
while True:
    r = res_queue.get()
    if r is not None: res.append(r)
    else: break
    if time() - start > 60: 
        print("time broken")
        break
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
# DQN (Fully informed)
# =============================================================================


import torch
from nlp2020.agents.dqn_agent import DQN_agent
import torch.multiprocessing as mp
from tqdm import tqdm

env = gym.make('nlp2020:nnlpDungeon-v0')

N_S = env.observation_space.n
N_A = env.action_space.n        
TARGET_UPDATE = 25

agent = DQN_agent(N_S, N_A)
num_missions = 20
num_episodes = 5000

best_mission_num = np.zeros(num_episodes)
rewards = np.zeros(num_episodes)
loop = tqdm(range(num_episodes))
for i_episode in loop:
    # Initialize the environment and state
    state = env.reset()
    cum_reward = 0
    for t in range(num_missions):
        # Select and perform an action
        
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
    
        cum_reward += reward
        
        # Observe new state
        if not done: next_state = state
        else: next_state = None

        # Store the transition in memory
        agent.update(i_episode, state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        agent.optimize_model()
        if done:
            best_mission_num[i_episode] = t
            rewards[i_episode] = cum_reward
            break
        
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        agent.target_net.load_state_dict(agent.policy_net.state_dict())


plt.subplot(1, 2, 1)

a,b = np.unique(best_mission_num,return_counts=True)
plt.bar(a,b/b.sum())
plt.xticks(range(num_missions),range(1,num_missions+1))
plt.title(f"Max epochs = {num_episodes}")

plt.subplot(1, 2, 2)

plt.plot(rewards)
plt.title(f"Max epochs = {num_episodes}")










































