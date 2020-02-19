! pip install --upgrade git+https://MichelangeloConserva:NLP_project_2020@github.com/MichelangeloConserva/NLP_project_2020.git


import gym
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm
from collections import Counter
from itertools import count

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.utils import smooth
from nlp2020.train_test_functions import train1, test1

"""
Without the grid we define the episode as a maximum of 10 missions. 
If the agents dies then the episode ends.
"""

# Hyperparameters
n_mission_per_episode = 10   # Default
n_equip_can_take = 2         # Default
n_trials = 5
long_episode_count = 400
short_episode_count = 50
env = gym.make('nlp2020:nnlpDungeon-v0')

# Create environments, agents and storing array
algs = {}
algs[RandomAgent(env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
                                         np.zeros((n_trials,long_episode_count)),
                                         train1, test1, "red", long_episode_count)

algs[DQN_agent(env.observation_space.n,env.action_space.n)] = (
    gym.make('nlp2020:nnlpDungeon-v0'), np.zeros((n_trials,short_episode_count)),
     train1, test1, "blue", short_episode_count)

env = gym.make('nlp2020:nnlpDungeon-v0')
env.is_fully_informed(False)
algs[DQN_agent(env.observation_space.n,env.action_space.n, fully_informed=False)] = (
    env, np.zeros((n_trials,short_episode_count)), train1, test1, "cyan",
     short_episode_count)

algs[ACER_agent(env.observation_space.n,env.action_space.n)] = (
    gym.make('nlp2020:nnlpDungeon-v0'),  np.zeros((n_trials,long_episode_count)),
     train1, test1, "green", long_episode_count)


                                        
                                        
                                       
                                        
                                        
# Running the experiment
save_models = False
for agent,(env,rewards,train_func,_,_,episode_count) in algs.items():
    loop = tqdm(range(n_trials))
    for trial in loop:
        
        # Agent reset learning before starting another trial
        agent.reset()
        
        # Training loop for a certain number of episodes
        train_func(agent, env, loop, episode_count, rewards, trial)
    
    if save_models: agent.save_model() 


# TRAINING PERFORMANCE
for agent,(env,rewards,_,_,col,_) in algs.items():
    cut = 20
    m = smooth(rewards.mean(0))[cut:]
    s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)
plt.hlines(0, 0, long_episode_count, color = "chocolate", linestyles="--")
plt.hlines(-n_mission_per_episode, 0, long_episode_count, color = "chocolate", linestyles="--")
plt.ylim(-n_mission_per_episode-0.5, 0.5)
plt.legend()
plt.show()



# TESTING PERFORMANCE
from itertools import count
n_test_trials = 5000
test_trials = {}

for agent,(env,_,_,test_func,_,_) in algs.items():
    test_trials[agent.name] = np.zeros(n_test_trials, dtype = int)
    
    loop = tqdm(range(n_test_trials))
    loop.set_description(f"{agent.name}")
    loop.refresh()    
    for trial in loop:    
        test_func(agent, env, trial, test_trials)


# Multi bars plot
spacing = np.linspace(-1,1, len(algs))
width = spacing[1] - spacing[0]
missions = np.arange(n_mission_per_episode*4, step = 4)
for (i,(agent,(_,_,_,_,col,_))) in enumerate(algs.items()):

    c = Counter(test_trials[agent.name])
    counts = [c[j]/n_test_trials for j in range(n_mission_per_episode)]
    
    plt.bar(missions + spacing[i], 
            counts, width, label = agent.name, color = col, edgecolor="black")
    
plt.xlabel("Consecutive mission, i.e. length of the episode")
plt.xticks(missions,range(1,n_mission_per_episode+1))
plt.legend()

    




























































# =============================================================================
# OLD VERSION THAT CAN BE USED FOR TESTING
# =============================================================================

import gym
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm
from collections import Counter
from itertools import count

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.utils import smooth


# Hyperparameters
n_mission_per_episode = 10 
n_equip_can_take = 2
n_trials = 5
episode_count = 100000
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
algs[ACER_agent(env.observation_space.n,
                env.action_space.n)] = (gym.make('nlp2020:nnlpDungeon-v0'),
                                        np.zeros((n_trials,episode_count)),
                                        "green")

# Running the experiment
loop = tqdm(range(n_trials))
for trial in loop:
    for agent,(env,rewards,_) in algs.items():
        # loop.set_description("%s, episode %9.d/%d" % (agent.name, i, episode_count))
        # loop.refresh()        
        
        # Agent reset learning before starting another trial
        agent.reset()
        
        for i in range(episode_count):
            if i % (episode_count // 5 - 1) == 0:
                loop.set_description(f"{agent.name}, episode loop {int(round(i/episode_count,2)*100)}%")
                loop.refresh()
            
            
            # Start of the episode
            agent.start_episode()
            done = False; cum_reward = 0
            state = env.reset()
            
            while not done:
            
                    # New dungeon
                    agent.before_act()

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
















# =============================================================================
# REINFORCE DOESN'T WORK
# =============================================================================


import gym
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm
from collections import Counter
from itertools import count

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.utils import smooth

class REINFORCE(object):

  def __init__(self, name, number_of_arms, alpha, baseline = False):
    self._number_of_arms = number_of_arms
    self.name = name
    self.reset()
    self.alpha = alpha
    self.b = lambda : (self.R / self.t if self.t != 0 else 0) if baseline else 0

  def step(self, previous_action, reward):
    if not previous_action is None:
        self.update_values(previous_action, reward)
    p = self.policy(range(self._number_of_arms))
    return np.random.choice(range(self._number_of_arms), p = p)

  def reset(self):
    self.H_a = np.zeros(self._number_of_arms)
    self.R = 0
    self.t = 0

  def update_values(self, previous_action, reward):
    p_t = self.policy(range(self._number_of_arms))

    c = self.alpha * (reward - self.b())
    self.H_a[previous_action] += c * (1 - p_t[previous_action])
    self.H_a[self.not_index(previous_action)] -= c *\
                                  (p_t[self.not_index(previous_action)])
    self.R += reward
    self.t += 1


  def policy(self, action):
    return np.exp(self.H_a[action]) / sum(np.exp(self.H_a))
  
  def not_index(self, action):
    return np.arange(self._number_of_arms) != action


# Hyperparameters
n_mission_per_episode = 10
n_equip_can_take = 2
n_trials = 5
episode_count = 20000 * 3
env = gym.make('nlp2020:nnlpDungeon-v0')


agent = REINFORCE("REINFORCE", env.action_space.n, 0.25, baseline = True)
rewards = np.zeros((n_trials,episode_count))
col = "black"

# Running the experiment
loop = tqdm(range(n_trials))
for trial in loop:
        # loop.set_description("%s, episode %9.d/%d" % (agent.name, i, episode_count))
        # loop.refresh()        
        
        # Agent reset learning before starting another trial
        agent.reset()
        
        for i in range(episode_count):
            if i % (episode_count // 5 - 1) == 0:
                loop.set_description(f"{agent.name}, episode loop {int(round(i/episode_count,2)*100)}%")
                loop.refresh()
            
            # Start of the episode
            # agent.start_episode()
            done = False; cum_reward = 0; previous_action = None; reward = 0;
            state = env.reset()

            while not done:

                action = agent.step(previous_action, reward)
                
                # Action perform
                next_state, reward, done, _ = env.step(action)
                cum_reward += reward
    
                # Observe new state
                if not done: next_state = state
                else: next_state = None   
                                      
        
             # End of the episode
            rewards[trial, i] = cum_reward
            # agent.end_episode()


# TRAINING PERFORMANCE
cut = 20
m = smooth(rewards.mean(0))[cut:]
s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
line = plt.plot(m, alpha=0.7, label=agent.name,
                  color=col, lw=3)[0]
plt.fill_between(range(len(m)), m + s, m - s,
                    color=line.get_color(), alpha=0.2)
plt.legend()
plt.show()















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
# algs[DQN_agent(4,
#                env.action_space.n)] = (gym.make('gym:CartPole-v0'),
#                                          np.zeros((n_trials,episode_count)),
#                                          "blue")

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

