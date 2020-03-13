from itertools import count
from collections import Counter

import numpy as np
import pickle

from nlp2020.utils import create_iterator

def train1(agent,env, loop, episode_count, rewards, trial):
    # Agent reset learning before starting another trial
    agent.reset()
    
    for i in range(episode_count):
        if i % 20 == 0:
            loop.set_description(f"{agent.name}, inn loop {int(round(i/episode_count,2)*100)}%")
            loop.refresh()
        
        # Start of the episode
        agent.start_episode()
        done = False; cum_reward = 0
        state = env.reset(agent.nlp)
        
        
        agent.before_act()

        # Action selection
        action = agent.act(state)
        
        # Action perform
        next_state, reward, done, _ = env.step(action, agent.nlp)
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



def test1(agent, env, trial, test_trials):

    # New dungeon
    state = env.reset(agent.nlp)
    done = False

    # Action selection
    action = agent.act(state, test = True)
    
    # Action perform
    next_state, reward, done, _ = env.step(action, agent.nlp)

    # Move to the next state
    state = next_state  

    # Storing number of consecutive missions
    test_trials[agent.name][trial] = reward
    
    
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
        
        if not done and \
           len(trial_rew) > 1000 and\
               np.mean(trial_rew[-1000:])>0.3 and \
           np.mean(trial_rew[-1000:-100]) - np.mean(trial_rew[-100:]) < 0.0001:
               
            agent.model = agent.model.eval()
            _, test_iterator, _, LABEL, TEXT = create_iterator("cuda", 128, int(2e3))
            
            n_test_trials = 10
            rs = []
            for _ in range(n_test_trials):
                for batch in test_iterator:
                    rs += agent.act_and_train(batch, test = True)
            agent.model = agent.model.train()
    
            with open("./logs_nlp2020/trial_"+agent.name+\
                      ".pickle", "wb") as f: pickle.dump(rs, f)
                
            done = True
            print(agent.name, "Performance for testing stored")
        
    return trial_rew
    