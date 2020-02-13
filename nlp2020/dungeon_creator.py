import numpy as np

class DungeonCreator():
    def __init__(self, effective_matrix):
        self.effective_matrix = effective_matrix
        self.num_of_dungeon, self.n_equip = effective_matrix.shape
        self.monsters = ["Wumpus","Wolf"]
        
        # At the moment dung type and living monster are the same
        self.dung_type = np.random.randint(self.num_of_dungeon)
        self.grid_width, self.grid_height = 3,4
        
        
        self.create_dung()
        
        
        # if np.random.random() < self.effective_matrix[dung_type][equipement_selected].sum():
        #     return +1
        # return -1
    
    def create_dung(self):
        # FOR NOW just 4x4
        self.living_monster = self.monsters[self.dung_type] 
        self.grid_world = np.array([["You", "None", "None", "None"],
                                   ["None", "None", self.living_monster, "None"],
                                   ["Rock", "None", "None", "Exit"]])
        self.your_pos = np.array((0,0))
        
        # Visual RGB Feature
        self.change_visual_map()     
    
    def change_visual_map(self):
        # Visual RGB Feature
        self.visual_features = np.zeros((self.grid_width, self.grid_height, 3)) + 1
        for i in range(self.grid_width):
            for k in range(self.grid_height):
                if self.grid_world[i,k] == "You":
                    self.visual_features[i,k,0] = 0
                elif self.grid_world[i,k] == "Rock":
                    self.visual_features[i,k] = 0
                elif self.grid_world[i,k] in self.monsters:
                    self.visual_features[i,k,1] = 0  
                elif self.grid_world[i,k] == "Exit":
                    self.visual_features[i,k,1] = 1 
                    self.visual_features[i,k,0] = 0.2    
                    self.visual_features[i,k,2] = 0.2    
        
        
    
    def check_for_bumping(self,new_pos):
        
        if new_pos[0] < 0 or new_pos[0] >= self.grid_width or\
            new_pos[1] < 0 or  new_pos[1] >= self.grid_height or\
            self.grid_world[new_pos[0], new_pos[1]] == "Rock":
            return True
        return False
    
    
    def kill_the_monster(self,new_pos):
            
        # Check if you can kill the monster
        if np.random.random() < self.effective_matrix[self.dung_type][self.equipement_selected].sum():
            return True  # You killed the monster
        return False      # You are dead    
                    
        
        
    def move(self, new_pos):
        self.grid_world[self.your_pos[0],self.your_pos[1]] = "None"
        self.grid_world[new_pos[0], new_pos[1]] = "You"    
        self.your_pos = new_pos
        self.change_visual_map()

    
    def reset(self):
        self.create_dung()
    
    
    
    def receiving_action(self,action):
        
        if action == 0:    # GOING UP
            new_pos = self.your_pos + [-1,0]
        elif action == 1:  # GOING RIGHT
            new_pos = self.your_pos + [0,1]
        elif action == 2:  # GOING DOWN
            new_pos = self.your_pos + [1,0]
        elif action == 3:  # GOING LEFT
            new_pos = self.your_pos + [0,-1]
        else:
            raise ValueError("action is outside of range 4")
            
        reward = -1
        
        # You bumped into a wall
        if self.check_for_bumping(new_pos): 
            return reward, False

        # Fighting a monster
        if self.grid_world[new_pos[0], new_pos[1]] == self.living_monster:
            
            victorious = self.kill_the_monster(new_pos)
            
            if victorious:
                reward = 100; self.move(new_pos)
            else: return -100, True
            
        # Reaching the exit
        if self.grid_world[new_pos[0], new_pos[1]] == "Exit":
            return reward, True
            
        # No special encouter, just moving around
        self.move(new_pos)
        return reward, False
            