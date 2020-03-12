from itertools import count
import numpy as np

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
    
    
def train2(agent, loop, n_trials, epochs, train_iterator, acc_hist, rewards, trial):
    best_valid_loss = float('inf')

    trial_rew = []
    for e in range(epochs):
        
        train_loss_SL = 0
        train_acc_SL = 0        
        
        for batch in train_iterator:
            loss_SL, acc_SL, r = agent.act_and_train(batch)
            
            train_acc_SL += acc_SL
            train_loss_SL += loss_SL
            trial_rew += r

        train_loss_SL /= len(train_iterator) 
        train_acc_SL  /= len(train_iterator)    
        
        acc_hist[trial, e] = train_acc_SL
        if agent.nlp:
            loop.set_description(f'{agent.name} | Epoch: {e} | Train Loss: {train_loss_SL:.3f} | Train Acc: {train_acc_SL*100:.2f}%'+\
                                   f'| Mean recent rewards: {np.mean(trial_rew[:-100]):.3f}')
        else:
            loop.set_description(f'{agent.name} | Epoch: {e}'+\
                   f'| Mean recent rewards: {np.mean(trial_rew[:-100]):.3f}')
            
            
    return trial_rew
    
    
    
        
        
        
        