from itertools import count

def train1(agent, env, loop, episode_count, rewards, trial):
    # Agent reset learning before starting another trial
    agent.reset()
    
    for i in range(episode_count):
        if i % (episode_count // 5 - 1) == 0:
            loop.set_description(f"{agent.name}, inn loop {int(round(i/episode_count,2)*100)}%")
            loop.refresh()
        
        
        # Start of the episode
        agent.start_episode()
        done = False; cum_reward = 0
        state = env.reset()
        
        while not done:
        
                agent.before_act()
    
                # Action selection
                action = agent.act(state)
                
                # Action perform
                next_state, reward, done, _ = env.step(action)
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
    state = env.reset()
    done = False
    for t in count():

        # Action selection
        action = agent.act(state, test = True)
        
        # Action perform
        next_state, reward, done, _ = env.step(action)

        # Move to the next state
        state = next_state  
        
        if done: break

    # Storing number of consecutive missions
    test_trials[agent.name][trial] = t
    
    
    
    
    
    
    
    
        
        
        
        
        
        
        