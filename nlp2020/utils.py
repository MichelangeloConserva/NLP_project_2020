import numpy as np
import matplotlib.pyplot as plt
import re
import torchtext
import torch

from collections import Counter
from nltk.corpus import stopwords

from nlp2020.dung_descr_score import dungeon_description_generator

try: stopwords.words('english')
except:
    import nltk
    nltk.download('stopwords')
    stopwords.words('english')
    
    
def smooth(array, smoothing_horizon=100., initial_value=0.):
  """Smoothing function for plotting. Credit to Deep Mind Lectures at UCL"""
  smoothed_array = []
  value = initial_value
  b = 1./smoothing_horizon
  m = 1.
  for x in array:
    m *= 1. - b
    lr = b/(1 - m)
    value += lr*(x - value)
    smoothed_array.append(value)
  return np.array(smoothed_array)


def multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials):
    # Multi bars plot
    spacing = np.linspace(-1,1, len(algs))
    width = spacing[1] - spacing[0]
    missions = np.arange((n_mission_per_episode+1)*4, step = 4)
    for (i,(_,(agent,_,_,_,_,col,_))) in enumerate(algs.items()):
        c = Counter(test_trials[agent.name])
        counts = np.zeros(n_mission_per_episode+1)
        for k,v in c.items(): counts[k] = v/n_test_trials

        assert round(sum(counts),5) == 1
        plt.bar(missions + spacing[i], 
                counts, width, label = agent.name, color = col, edgecolor="black")
        
    plt.xlabel("Consecutive mission, i.e. length of the episode")
    plt.xticks(missions,range(n_mission_per_episode+1))
    plt.legend()
    
def tokenize(sentence,max_sentence_length=95):
    
    # Remove punctuations and numbers
    sentence =  re.sub('[^a-zA-Z]', ' ', sentence)[:-1].lower()

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence).split(" ")        
    
    sentence = [word for word in sentence if (word not in stopwords.words('english'))]
    
    # token = pad_sequences(sentence, maxlen=max_sentence_length, 
    #                       dtype="long", value=0, truncating="post", padding="post")
    token = sentence + [""] * (max_sentence_length - len(sentence))
    
    return token

def onehot_to_int(v): return v.index(1)
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_dataset(n = 2000):
    train_x = []
    train_y_temp = []
    train_y = []
    for i in range(n):
      description, label, _ = dungeon_description_generator()
      train_x.append(description)
      train_y_temp.append(label)
    for i in train_y_temp: train_y.append(onehot_to_int(i.tolist()))
    
    val_x = []
    val_y_temp = []
    val_y = []
    for i in range(n):
      description, label, _ = dungeon_description_generator()
      val_x.append(description)
      val_y_temp.append(label)
    for i in val_y_temp: val_y.append(onehot_to_int(i.tolist()))
    
    test_x = []
    test_y_temp = []
    test_y = []
    for i in range(n):
      description, label, _ = dungeon_description_generator()
      test_x.append(description)
      test_y_temp.append(label)
    for i in test_y_temp: test_y.append(onehot_to_int(i.tolist()))

    return train_x, val_x, test_x, train_y, val_y, test_y

def ListToTorchtext(train_x, val_x, test_x, train_y, val_y, test_y, datafields):
  train = []
  for i,line in enumerate(train_x):
    doc = line.split()
    train.append(torchtext.data.Example.fromlist([doc, train_y[i]], datafields))
  val = []
  for i,line in enumerate(val_x):
    doc = line.split()
    val.append(torchtext.data.Example.fromlist([doc, val_y[i]], datafields))
  test = []
  for i,line in enumerate(test_x):
    doc = line.split()
    test.append(torchtext.data.Example.fromlist([doc, test_y[i]], datafields))
  return torchtext.data.Dataset(train, datafields), torchtext.data.Dataset(val, datafields), torchtext.data.Dataset(test, datafields)


def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])






