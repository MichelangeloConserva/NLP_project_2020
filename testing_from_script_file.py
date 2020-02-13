! pip install --upgrade git+https://MichelangeloConserva:NLP_project_2020@github.com/MichelangeloConserva/NLP_project_2020.git

# For selecting equipments we use multilabel classification style
# i.e. we put a sigmoid on the final layer and take highest k



import gym
import numpy as np
import matplotlib.pyplot as plt

from nlp2020.agents.random_agent import RandomAgent






"""
Without the grid we define the episode as a maximum of 10 missions. 
If the agents dies then the episode ends.
"""


# =============================================================================
# Random agent
# =============================================================================
random_agent = RandomAgent(env.action_space.n)
creator = DungeonCreator(effectivness_matrix, n_equip_can_take)
env = gym.make('nlp2020:nnlpDungeon-v0')

n_mission_per_episode = 10 

done = False
reward = 0
episode_count = 100
rewards = np.zeros(episode_count)
for i in range(episode_count):
    ob = env.reset()
    
    
    cum_reward = 0
    for _ in range(n_mission_per_episode):
        action = random_agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        cum_reward += reward
        
        if done: break
            
    rewards[i] += cum_reward
    
rewards = rewards / episode_count
plt.plot(rewards)

env.close()

# =============================================================================
# AC3 agent (Fully informed)
# =============================================================================

from nlp2020.agents.ac3_agent import Net, SharedAdam, v_wrap, Worker
import torch.multiprocessing as mp
    

creator = DungeonCreator(effectivness_matrix, n_equip_can_take)
env = gym.make('nlp2020:nnlpDungeon-v0', 
               dungeon_creator = creator)

N_S = env.observation_space.n
N_A = env.action_space.n        
num_missions = 20
gnet = Net(N_S, N_A)        # global network
gnet.share_memory()         # share the global parameters in multiprocessing
opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

# parallel training
workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, 
                  creator, num_missions, max_ep = 4000) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []                    # record episode reward to plot
while True:
    r = res_queue.get()
    if r is not None:
        res.append(r)
    else:
        break
[w.join() for w in workers]

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
# AC3 agent (not informed)
# =============================================================================

from nlp2020.agents.ac3_agent import Net, SharedAdam, v_wrap, Worker
import torch.multiprocessing as mp
    
N_S = env.observation_space.n
N_A = env.action_space.n        
num_missions = 20
creator = DungeonCreator(effectivness_matrix, n_equip_can_take)
env = gym.make('nlp2020:nnlpDungeon-v0', 
               dungeon_creator = creator)

gnet = Net(N_S, N_A)        # global network
gnet.share_memory()         # share the global parameters in multiprocessing
opt = SharedAdam(gnet.parameters(), lr=0.0001)      # global optimizer
global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

# parallel training
workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, 
                  creator, num_missions, max_ep = 1000) for i in range(mp.cpu_count())]
[w.start() for w in workers]
res = []                    # record episode reward to plot
while True:
    r = res_queue.get()
    if r is not None:
        res.append(r)
    else:
        break
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

creator = DungeonCreator(effectivness_matrix, n_equip_can_take)
env = gym.make('nlp2020:nnlpDungeon-v0')

N_S = env.observation_space.n
N_A = env.action_space.n        
TARGET_UPDATE = 25

agent = DQN_agent(N_S, N_A)
num_missions = 20
num_episodes = 10000

best_mission_num = np.zeros(num_episodes)
rewards = np.zeros(num_episodes)
loop = tqdm(range(num_episodes))
for i_episode in loop:
    # Initialize the environment and state
    state = torch.tensor(env.reset(), dtype = torch.float, device = agent.device).view(1,-1)
    cum_reward = 0
    for t in range(num_missions):
        # Select and perform an action
        
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action.item())
        next_state = torch.tensor(next_state, dtype = torch.float, device = agent.device)
    
        reward = torch.tensor([reward], device=agent.device)
        cum_reward += reward
        
        # Observe new state
        if not done: next_state = state
        else: next_state = None

        # Store the transition in memory
        agent.memory.push(state, action, next_state, reward)

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










































