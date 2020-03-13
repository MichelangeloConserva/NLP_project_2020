import numpy as np
import matplotlib.pyplot as plt
import re
import torchtext
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from collections import Counter
from nltk.corpus import stopwords
from scipy.stats import invgamma
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def create_dataset(n = 2000):
    onehot_to_int = lambda v : v.index(1)
    
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


def create_iterator(device, BATCH_SIZE, N = 5000):

    train_x, val_x, test_x, train_y, val_y, test_y = create_dataset(N)
    
    TEXT = data.Field(tokenize = tokenize)
    LABEL = data.LabelField()
    datafields = [('text', TEXT), ('label', LABEL)]
    TrainData, ValData, TestData = ListToTorchtext(train_x, val_x, test_x, train_y, val_y, test_y, datafields)
    
    TEXT.build_vocab(TrainData)
    LABEL.build_vocab(train_y)
    
    # print(LABEL.vocab.stoi)
    # print(len(TEXT.vocab))
    
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (TrainData, ValData, TestData), 
        batch_size = BATCH_SIZE, 
        device = device,
        shuffle = True,
        sort=False)
    valid_iterator = torchtext.data.Iterator(
        ValData,
        device=device,
        batch_size=128,
        repeat=False,
        train=False,
        sort=False)

    return  train_iterator, valid_iterator, test_iterator, LABEL, TEXT
