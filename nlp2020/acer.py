import gym, torch
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)
from tqdm import tqdm

from nlp2020.agents.random_agent  import RandomAgent
from nlp2020.agents.dqn_agent     import DQN_agent
from nlp2020.agents.acer_agent    import ACER_agent
from nlp2020.utils                import smooth, multi_bar_plot
from nlp2020.train_test_functions import train1, test1

# Hyperparameters
n_mission_per_episode   = 10    # Every episode is made of consecutive missions
n_equip_can_take        = 1     # Equipement the explores has for every mission
n_trials                = 1     # Trials for estimating performance (training) 
n_test_trials           = 1000   # Trials for estimating performance (testing)   
buffer_size             = 1000  # Buffer size for memory cells of the algorithms
batch_size              = 128
episode_before_train    = batch_size + 1
episode_count           = int(1e4)  # Number of episodes for training
# training_time           = 5 * 60 
NNLP_env= env           = gym.make('nlp2020:nnlpDungeon-v0')
NLP_env                 = gym.make('nlp2020:nlpDungeon-v0')

# Setting equip
NNLP_env.set_num_equip(n_equip_can_take)
NLP_env.set_num_equip(n_equip_can_take)
env.set_num_equip(n_equip_can_take)

algs = {}
# Create the data structure that contains all the stuff for train and test
"""
{agent : (environment, array for storing rewards, train function,
          test_function, color for plots, number of episode to run)}
"""

# ACER NLP FULLY INFORMED
# agent = ACER_agent(env.observation_space.n, env.action_space.n,
#                 fully_informed       = True,
#                 nlp                  = True,
#                 learning_rate        = 0.002,
#                 gamma                = 0.98,
#                 buffer_limit         = buffer_size , 
#                 rollout_len          = 2 ,
#                 batch_size           = batch_size,     
#                 c                    = 1.0, 
#                 max_sentence_length  = 100,
#                 episode_before_train = episode_before_train)
# algs[agent.name] = (agent, NLP_env, np.zeros((n_trials,episode_count)),
#                 train1, test1, "lawngreen", episode_count)   
      
# ACER NOT NLP FULLY INFORMED
agent = ACER_agent(env.observation_space.n, env.action_space.n,
                fully_informed       = True,
                nlp                  = False,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size, 
                rollout_len          = 2,
                batch_size           = batch_size,
                c                    = 1.0, 
                max_sentence_length  = 100,
                episode_before_train = episode_before_train
                )
algs[agent.name] = (agent, NNLP_env, np.zeros((n_trials,episode_count)),
                    train1, test1, "green", episode_count)   
      
# ACER NOT FULLY INFORMED
agent = ACER_agent(env.observation_space.n, env.action_space.n,
                fully_informed       = False,
                nlp                  = False,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size, 
                rollout_len          = 2,
                batch_size           = batch_size,
                c                    = 1.0,
                max_sentence_length  = 100,
                episode_before_train = episode_before_train         
                )
algs[agent.name] = (agent, NNLP_env, np.zeros((n_trials,episode_count)),
               train1, test1, "palegreen", episode_count)  

# RANDOM AGENT
algs["Random"] = (RandomAgent(env.action_space.n), NNLP_env, np.zeros((n_trials,episode_count)),
          train1, test1, "red", episode_count) 

# Running the experiment
save_models = True;  load = False
for _,(agent,env,rewards,train_func,_,_,episode_count) in algs.items():
    loop = tqdm(range(n_trials))
    for trial in loop:

        agent.reset() # Agent reset learning before starting another trial
        if load: agent.load_model()
        
        # Training loop for a certain number of episodes
        train_func(agent, env, loop, episode_count, rewards, trial)
    
    if save_models: agent.save_model() 

# TRAINING PERFORMANCE
for _,(agent,env,rewards,_,_,col,_) in algs.items():
    cut = 20
    m = smooth(rewards.mean(0))[cut:]
    s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
    line = plt.plot(m, alpha=0.7, label=agent.name,
                      color=col, lw=3)[0]
    plt.fill_between(range(len(m)), m + s, m - s,
                        color=line.get_color(), alpha=0.2)
plt.hlines(0, 0, episode_count, color = "chocolate", linestyles="--")
plt.hlines(-n_mission_per_episode, 0, episode_count, color = "chocolate", linestyles="--")
plt.ylim(-n_mission_per_episode-0.5, 0.5)
plt.legend(); plt.show()

# TESTING PERFORMANCE
test_trials = {}
for _,(agent, env,_,_,test_func,_,_) in algs.items():
    test_trials[agent.name] = np.zeros(n_test_trials, dtype = int)
    loop = tqdm(range(n_test_trials), desc = f"{agent.name}"); loop.refresh()  
    for trial in loop: test_func(agent, env, trial, test_trials)
multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials)



if False:

# =============================================================================
# DEBUGGER
# =============================================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nnlp_ninf = algs["ACERAgent_NotInformed_NNLP"][0]
    nnlp_inf  = algs["ACERAgent_FullyInformed_NNLP"][0]
    nlp_inf   = algs["ACERAgent_FullyInformed_NLP"][0]
    nlp_lay = nlp_inf.model.NLP
    
    NLP_env.reset()
    det = torch.tensor(NLP_env.dungeon_creator.dung_type).to(device)
    # desc = torch.tensor(nlp_inf.tokenize(NLP_env.dungeon_creator.dungeon_description)).to(device)
    ndet = torch.zeros(5).to(device)
    with torch.no_grad():
        print(det)
        print("nnlp_ninf",nnlp_ninf.model.pi(ndet).cpu().numpy(),
              nnlp_ninf.model.pi(ndet).cpu().numpy().argmax())
        print("nnlp_inf",nnlp_inf.model.pi(det.float()).cpu().numpy(),
              nnlp_inf.model.pi(det.float()).cpu().numpy().argmax())
        # print("nlp_inf",nlp_inf.model.pi(desc).cpu().numpy(),
        #       nlp_inf.model.pi(desc).cpu().numpy().argmax(), "\n")
        # print("nlp layer", nlp_lay(desc).cpu().numpy().round(2))
    



# # agent = algs["DQNAgent_FullyInformed_NLP"]
# agent = list(algs.keys())[0]

# model = list(agent.model.children())[0]


# state = NLP_env.reset()
# print("Actual dungeon:", NLP_env.dungeon_creator.dung_type.argmax())
# print("Predicted dungeon:", model(torch.tensor(agent.tokenize(state)).float()))

# state = NLP_env.reset()
# print(state)
# torch.tensor(agent.tokenize(state)).sum()
















