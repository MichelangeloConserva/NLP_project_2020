! pip install --upgrade git+https://MichelangeloConserva:NLP_project_2020@github.com/MichelangeloConserva/NLP_project_2020.git@with_grid

# For selecting equipments we use multilabel classification style
# i.e. we put a sigmoid on the final layer and take highest k



import gym
import numpy as np
import matplotlib.pyplot as plt

from nlp2020.dungeon_creator import DungeonCreator 
from nlp2020.agents.random_agent import RandomAgent

    
    
def training_loop(agent, render = False):
    
    episode_count = 100
    reward = 0
    done = False
    
    rewards = np.zeros(episode_count)
    for i in range(episode_count):
        
        ob = env.reset()
        
        # Equipment selection phase
        selection = random_agent.act(ob, reward, done, equipment_selection = True)
        env.store_selection(selection, equipment)
    
        if render: env.render()
            
        cum_reward = 0
        while True:
            
            action = random_agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            
            cum_reward += reward
            
            if render: env.render()
            if done: break
    
        rewards[i] += cum_reward
        
    env.close()
    
    
    rewards = rewards / episode_count
    plt.plot(rewards)
        
    
equipment = ["sword", "bow", "water", "pickaxe"]
n_env = 2
action_space_dim = len(equipment)
n_equip_can_take = 2
effectivness_matrix = np.array(
    [[0.8, 0.1, 0.05, 0.05],
     [0.2, 0.1, 0.2, 0.5]])
creator = DungeonCreator(effectivness_matrix)





# =============================================================================
# Random Agent
# =============================================================================

env = gym.make('nlp2020:nnlpDungeon-v0', 
               dungeon_creator = creator)

random_agent = RandomAgent(action_space_dim, n_equip_can_take)
training_loop(random_agent)

        














