from collections import Counter

import numpy as np


def train_f(agent, loop, n_trials, epochs, train_iterator, acc_hist, rewards, trial):

    mean_last_r = 0
    
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
        
        r = np.array(r)
        if agent.nlp:
            tlta = f"Train Loss: {train_loss_SL:.3f} | Train Acc: {train_acc_SL*100:.2f}%" \
                    if train_acc_SL != 0 else ""
            loop.set_description(f'{agent.name} | Epoch: {e} |'+tlta+\
                                   f'| Mean recent rewards: {mean_last_r:.3f}' +\
                                   f"| Last actions: {dict(Counter(actions.tolist()).most_common())} |"+\
                   f"Positive rewards: {(r > 0).mean():.2f}%")
        else:
            loop.set_description(f'{agent.name} | Epoch: {e}'+\
                   f'| Mean recent rewards: {mean_last_r:.3f}'+\
                   f"| Last actions: {dict(Counter(actions.tolist()).most_common())}"+\
                   f"Positive rewards: {(r > 0).mean():.2f}%")
        
        mean_last_r = np.mean(r) 
    return trial_rew
    