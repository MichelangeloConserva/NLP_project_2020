import gym, torch, torchtext
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
np.set_printoptions(precision=3, suppress=1)

from tqdm import tqdm
from torchtext import data

from nlp2020.agents.random_agent import RandomAgent
from nlp2020.agents.dqn_agent import DQN_agent
from nlp2020.agents.acer_agent import ACER_agent
from nlp2020.utils import smooth, tokenize, count_parameters, create_dataset, categorical_accuracy, ListToTorchtext
from nlp2020.train_test_functions import train1, test1
from nlp2020.architectures import CNN, ActorCritic

SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 256
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [2,3,4]
DROPOUT = 0.5
device = torch.device("cuda")

train_x, val_x, test_x, train_y, val_y, test_y = create_dataset()

TEXT = data.Field(tokenize = tokenize)
LABEL = data.LabelField()
datafields = [('text', TEXT), ('label', LABEL)]
TrainData, ValData, TestData = ListToTorchtext(train_x, val_x, test_x, train_y, val_y, test_y, datafields)

TEXT.build_vocab(TrainData)
LABEL.build_vocab(train_y)

print(LABEL.vocab.stoi)
print(len(TEXT.vocab))

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (TrainData, ValData, TestData), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort=False)
valid_iterator = torchtext.data.Iterator(
    ValData,
    device=device,
    batch_size=128,
    repeat=False,
    train=False,
    sort=False)


INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

print(f'The model has {count_parameters(model):,} trainable parameters')

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)



optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        predictions = model(batch.text)
        loss = criterion(predictions, batch.label)
        acc = categorical_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    with torch.no_grad():
    
        for batch in iterator:
            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

N_EPOCHS = 50

best_valid_loss = float('inf')
loop = tqdm(range(N_EPOCHS))
for epoch in loop:

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut5-model.pt')
    
    loop.set_description(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%  ' +\
                         f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%  ')

class NLP_ActorCritic(nn.Module):

    def __init__(self, k, action_dim, NLP):
        nn.Module.__init__(self)   
        # self.NLP = NLP_NN_EASY(vocab_dim, k)
        self.NLP = NLP
        self.RL  = ActorCritic(k, action_dim)
    
    def pi(self, x, softmax_dim = 0): 
        if x.dim() != 2: 
            x = x.squeeze()
        return self.RL.pi(self.NLP(x))
    def q(self, x):                   
        if x.dim() != 2: 
            x = x.squeeze()
        return self.RL.q(self.NLP(x))

# Hyperparameters
n_mission_per_episode   = 10    # Every episode is made of consecutive missions
n_equip_can_take        = 1     # Equipement the explores has for every mission
n_trials                = 2     # Trials for estimating performance (training) 
n_test_trials           = 1000   # Trials for estimating performance (testing)   
buffer_size             = int(5e3)  # Buffer size for memory cells of the algorithms
batch_size              = 64
step_before_train    = batch_size + 1
episode_count           = int(1e3)  # Number of episodes for training
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
agent = ACER_agent(env.observation_space.n, env.action_space.n,
                fully_informed       = True,
                nlp                  = True,
                learning_rate        = 0.0002,
                gamma                = 0.98,
                buffer_limit         = buffer_size , 
                rollout_len          = 3,
                batch_size           = batch_size,     
                c                    = 1.0, 
                max_sentence_length  = 100,
                step_before_train = step_before_train)

agent.device = device
agent.model = NLP_ActorCritic(5, 7, model).to(device)
agent.tokenize = lambda s : TEXT.process([TEXT.tokenize(s)], device = device).T

algs[agent.name] = (agent, NLP_env, np.zeros((n_trials,episode_count)),
                train1, test1, "lime", episode_count) 

# RANDOM AGENT
algs["RandomAgent"] = (RandomAgent(env.action_space.n), NNLP_env, np.zeros((n_trials,episode_count)),
          train1, test1, "red", episode_count) 


# Running the experiment
save = False;  load = False; load_reward = False;
for _,(agent,env,rewards,train_func,_,_,episode_count) in algs.items():
    loop = tqdm(range(n_trials))
    for trial in loop:

        if agent.name == "RandomAgent":
              agent.reset() # Agent reset learning before starting another trial
        else: agent.model = NLP_ActorCritic(5, 7, model)
        if load: agent.load_model()
        
        # Training loop for a certain number of episodes
        train_func(agent, env, loop, episode_count, rewards, trial)
    
    if save and agent.name != "RandomAgent": agent.save_model(algs[agent.name][2]) 

    if load_reward:
        old = np.loadtxt("./logs_nlp2020/"+agent.name+".txt")
        if len(old.shape) == 1: old = old.reshape(1,-1)
        new = algs[agent.name][2]
        algs[agent.name][2] = np.hstack((old,new))


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


from nlp2020.utils import multi_bar_plot
# TESTING PERFORMANCE
test_trials = {}
for _,(agent, env,_,_,test_func,_,_) in algs.items():
    agent.model.eval()
    test_trials[agent.name] = np.zeros(n_test_trials, dtype = int)
    loop = tqdm(range(n_test_trials), desc = f"{agent.name}"); loop.refresh()  
    for trial in loop: test_func(agent, env, trial, test_trials)
multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials)

















