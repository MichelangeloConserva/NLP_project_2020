! pip install --upgrade git+https://MichelangeloConserva:NLP_project_2020@github.com/MichelangeloConserva/NLP_project_2020.git

# For selecting equipments we use multilabel classification style
# i.e. we put a sigmoid on the final layer and take highest k



import gym
import numpy as np

class DungeonCreator():
    def __init__(self, effective_matrix):
        self.effective_matrix = effective_matrix
        self.num_of_dungeon, self.n_equip = effective_matrix.shape
        
    def result(self, equipement_selected):
        dung_type = np.random.randint(self.num_of_dungeon)
        
        if np.random.random() < self.effective_matrix[dung_type][equipement_selected].sum():
            return +1
        return -1
    
equipment = ["sword", "bow", "water", "pickaxe"]
n_env = 2
action_space_dim = len(equipment)
n_equip_can_take = 2
effectivness_matrix = np.array(
    [[0.8, 0.1, 0.05, 0.05],
     [0.2, 0.1, 0.2, 0.5]])
creator = DungeonCreator(effectivness_matrix)





env = gym.make('nlp2020:nnlpDungeon-v0', 
               dungeon_creator = creator)


random_agent = RandomAgent(env.action_space)

episode_count = 100
reward = 0
done = False

for i in range(episode_count):
    ob = env.reset()
    
    
    
    while True:
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        if done:
            break
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

# Close the env and write monitor result info to disk
env.close()














