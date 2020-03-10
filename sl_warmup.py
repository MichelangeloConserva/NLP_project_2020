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
from nlp2020.dung_descr_score import dungeon_description_generator


# Hyperparameters
n_mission_per_episode   = 10    # Every episode is made of consecutive missions
n_equip_can_take        = 1     # Equipement the explores has for every mission
n_trials                = 2     # Trials for estimating performance (training) 
n_test_trials           = 100   # Trials for estimating performance (testing)   
buffer_size             = 1000  # Buffer size for memory cells of the algorithms
batch_size              = 64
episode_before_train    = batch_size + 1
episode_count           = int(1e3)  # Number of episodes for training
# training_time           = 5 * 60 
NNLP_env= env           = gym.make('nlp2020:nnlpDungeon-v0')
NLP_env                 = gym.make('nlp2020:nlpDungeon-v0')

# Setting equip
NNLP_env.set_num_equip(n_equip_can_take)
NLP_env.set_num_equip(n_equip_can_take)
env.set_num_equip(n_equip_can_take)

# ACER NLP FULLY INFORMED
agent = ACER_agent(env.observation_space.n, env.action_space.n,
                fully_informed       = True,
                nlp                  = True,
                learning_rate        = 0.002,
                gamma                = 0.98,
                buffer_limit         = buffer_size , 
                rollout_len          = 2 ,
                batch_size           = batch_size,     
                c                    = 1.0, 
                max_sentence_length  = 100,
                episode_before_train = episode_before_train)


# =============================================================================
# WARM UP
# =============================================================================
wu_instances = 1000

def int_to_onehot(n, n_classes):
    v = [0] * n_classes
    v[n] = 1
    return v

def onehot_to_int(v): return v.index(1)


y_train = np.zeros((wu_instances, 7))
x_train = np.zeros((wu_instances, 100))

for i in range(wu_instances):
  description, dung, _ = dungeon_description_generator()
  label = dung.argmax().astype(int)
  
  x_train[i,:] = agent.tokenize(description).squeeze()
  y_train[i] = label


device = "cuda"

model = agent.model
model.to(device)


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

x_data = np.hstack((x_train,y_train))
train_sampler = RandomSampler(x_data)
train_dataloader = DataLoader(x_data, sampler=train_sampler, batch_size=batch_size)

import torch
loss = torch.nn.CrossEntropyLoss()

model = agent.model
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), 0.01)

loss_values = []
epochs = 300
loop = tqdm(range(epochs))
for epoch_i in loop:
    
    total_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[:,:-7].to(device)
        b_labels = batch[:,-1].long().to(device)

        model.zero_grad()        
        pred = model.pi(b_input_ids)
        l = loss(pred, b_labels)
        total_loss += l.item()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    loop.set_description("Average training loss: {0:.2f}".format(avg_train_loss))
    loop.refresh()

plt.plot(loss_values)


with torch.no_grad():
    pred = model.pi(b_input_ids).cpu().numpy()[:10]

pred.argmax(1)
b_labels.cpu().numpy()[:10]
pred.argmax(1) == b_labels.cpu().numpy()[:10]






algs = {}
algs[agent.name] = (agent, NNLP_env, np.zeros((n_trials,episode_count)),
                    train1, test1, "green", episode_count)  
algs["Random"] = (RandomAgent(env.action_space.n), NNLP_env, np.zeros((n_trials,episode_count)),
          train1, test1, "red", episode_count) 

save_models = True;  load = False
for _,(agent,env,rewards,train_func,_,_,episode_count) in algs.items():
    loop = tqdm(range(n_trials))
    for trial in loop:

        # Forcing to cpu
        agent.reset() # Agent reset learning before starting another trial
        if load: agent.load_model()
        
        # Training loop for a certain number of episodes
        train_func(agent, env, loop, episode_count, rewards, trial)
    
    if save_models: agent.save_model() 





































