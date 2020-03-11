import numpy as np
import torch
import torchtext
import random
import matplotlib.pyplot as plt
import gym

from tqdm import tqdm
from torchtext import data

from nlp2020.dung_descr_score import dungeon_description_generator
from nlp2020.utils import tokenize, ListToTorchtext, ContextualDataset, NeuralBanditModel,NeuralLinearPosteriorSampling
from nlp2020.utils import ContextualBandit, multi_bar_plot, smooth


num_dung = 5
num_weap = 7

reward_win = 5
reward_die = -10


# =============================================================================
# Worst way possible to get the dict
# =============================================================================

N = 3000
x_train = []
y_train = []
for n in range(N):
    context, dung_identifier, probs_per_weapon = dungeon_description_generator()

    x_train.append(context)
    y_train.append(
    [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
    )
    
x_test = []
y_test = []
for n in range(N):
    context, dung_identifier, probs_per_weapon = dungeon_description_generator()

    x_test.append(context)
    y_test.append(
    [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
    )
    
x_val = []
y_val = []
for n in range(N):
    context, dung_identifier, probs_per_weapon = dungeon_description_generator()

    x_val.append(context)
    y_val.append(
    [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
    )


TEXT = data.Field(tokenize = tokenize)
LABEL = data.RawField()
datafields = [('text', TEXT), ('label', LABEL)]

TrainData, ValData, TestData = ListToTorchtext(x_train, x_val, x_test, y_train, y_val, y_test, datafields)

TEXT.build_vocab(TrainData)
# print(len(TEXT.vocab))s



# =============================================================================
# Actual script
# =============================================================================
SEED = 1234
MAX_VOCAB_SIZE = 25_000
BATCH_SIZE = 128
EMBEDDING_DIM = 125
N_FILTERS = 150
FILTER_SIZES = [2,3,4]
DROPOUT = 0.1
INPUT_DIM = len(TEXT.vocab)
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
device = torch.device("cuda")

env = gym.make('nlp2020:cbe-v0')
N = int(2e3)
dataset = np.zeros((N, 95 + 7))
for n in tqdm(range(N), desc = "Dataset creation"):
    env.reset()
    context = env.dungeon_description
    probs_per_weapon = env.probs_per_weapon
    
    context = TEXT.process([TEXT.tokenize(context)], device = "cpu").squeeze().numpy()
    
    r = [env.reward_die if random.random() > p_w else env.reward_win  for p_w in probs_per_weapon]
    dataset[n] = np.concatenate((context, np.array(r)))


algo =  NeuralLinearPosteriorSampling('NeuralLinear',
                                      vocab_size = len(TEXT.vocab),
                                      embedding_dim = EMBEDDING_DIM,
                                      filter_sizes = FILTER_SIZES,
                                      n_filters = N_FILTERS,
                                      dropout = 0.1,
                                      pad_idx = PAD_IDX)

algos = [algo]

# Create contextual bandit
cmab = ContextualBandit(95, 7)
cmab.feed_data(dataset)                                     

N = int(5e3)

h_actions = np.empty((0, len(algos)), float)
h_rewards = np.empty((0, len(algos)), float)
rand_rewards = np.zeros((N))

# Run the contextual bandit process
loop = tqdm(range(N), desc = algo.name)
algo.loop = loop

for ii in loop:
    
    i = np.random.randint(0,len(dataset))
    
    context = cmab.context(i).reshape(1,-1)
    action = algo.action(context) 
    reward  = cmab.reward(i, action)
  
    algo.update(context, action, reward)

    h_actions = np.vstack((h_actions, np.array(action)))
    h_rewards = np.vstack((h_rewards, np.array(reward)))

    r_a = np.random.randint(0,7)
    r = cmab.reward(i, r_a)
    rand_rewards[ii] = r


from collections import Counter
acts = []
algo.bnn.eval()
for _ in range(1000):
    acts.append(algo.action(context))
c = Counter(acts)
print(c)



cut = 20

rewards = h_rewards.reshape(1,-1)
col = "green"

m = smooth(rewards.mean(0))[cut:]
s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
line = plt.plot(m, alpha=0.7, label=algo.name,
                  color=col, lw=3)[0]
plt.fill_between(range(len(m)), m + s, m - s,
                    color=line.get_color(), alpha=0.2)
plt.hlines(env.reward_win, env.reward_win, N, color = "chocolate", linestyles="--")
plt.hlines(env.reward_die, env.reward_die, N, color = "chocolate", linestyles="--")
plt.ylim(env.reward_die-0.5, env.reward_win + 0.5)


rewards = rand_rewards.reshape(1,-1)
col = "red"

m = smooth(rewards.mean(0))[cut:]
s = (np.std(smooth(rewards.T).T, axis=0)/np.sqrt(len(rewards)))[cut:]
line = plt.plot(m, alpha=0.7, label=algo.name,
                  color=col, lw=3)[0]
plt.fill_between(range(len(m)), m + s, m - s,
                    color=line.get_color(), alpha=0.2)
plt.hlines(env.reward_win, env.reward_win, N, color = "chocolate", linestyles="--")
plt.hlines(env.reward_die, env.reward_die, N, color = "chocolate", linestyles="--")
plt.ylim(env.reward_die-0.5, env.reward_win + 0.5)
plt.legend(); plt.show()


if False:

    # Validation
    n_test_trials = 200
    
    test_trials = np.zeros(n_test_trials, dtype = int)
    model = algo.bnn
    model.eval()
        
    for i in tqdm(range(n_test_trials),desc="Testing"):
        context, dung_identifier, probs_per_weapon = dungeon_description_generator()
        context = TEXT.process([TEXT.tokenize(context)], device = "cpu").numpy().T
    
        action = algo.action(context) 
    
        r = [reward_die if random.random() < p_w else reward_win  for p_w in probs_per_weapon]
    
        test_trials[i] = r[action]
    
    
    from collections import Counter
    c = Counter(test_trials)
    c
    






















