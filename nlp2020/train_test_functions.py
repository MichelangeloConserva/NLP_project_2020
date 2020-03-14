from collections import Counter

import numpy as np
import pickle

from nlp2020.utils import create_iterator

def train_f(agent, loop, n_trials, epochs, train_iterator, acc_hist, rewards, trial):

    done = False
    
    trial_rew = []
    for e in range(epochs):
        
        train_loss_SL = 0
        train_acc_SL = 0        
        
        for batch in train_iterator:
            if "Random" not in agent.name:
                loss_SL, acc_SL, r, actions = agent.act_and_train(batch, return_actions = True)
            else:
                loss_SL, acc_SL, r = agent.act_and_train(batch); actions = np.array([])
               
            train_acc_SL += acc_SL
            train_loss_SL += loss_SL
            trial_rew += r

        train_loss_SL /= len(train_iterator) 
        train_acc_SL  /= len(train_iterator)    
        
        acc_hist[trial, e] = train_acc_SL
        if agent.nlp:
            tlta = f"Train Loss: {train_loss_SL:.3f} | Train Acc: {train_acc_SL*100:.2f}%" \
                    if train_acc_SL != 0 else ""
            loop.set_description(f'{agent.name} | Epoch: {e} |'+tlta+\
                                   f'| Mean recent rewards: {np.mean(trial_rew[:-100]):.3f}' +\
                                   f"| Last actions: {dict(Counter(actions.tolist()).most_common())}")
        else:
            loop.set_description(f'{agent.name} | Epoch: {e}'+\
                   f'| Mean recent rewards: {np.mean(trial_rew[:-100]):.3f}'+\
                   f"| Last actions: {dict(Counter(actions.tolist()).most_common())}")
        
    return trial_rew
    