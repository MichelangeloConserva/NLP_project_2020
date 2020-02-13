import numpy as np

class DungeonCreator():
    def __init__(self, effective_matrix, n_equip_can_take, fully_informed = True):
        self.effective_matrix = effective_matrix
        self.num_of_dungeon, self.n_equip = effective_matrix.shape
        self.n_equip_can_take = n_equip_can_take
        self.fully_informed = fully_informed
        
    def result(self, equipement_selected):
        if np.random.random() < self.effective_matrix[self.dung_type.argmax()][equipement_selected].sum():
            return +1, False
        return -1, True
    
    def reset(self):
        dung_type = np.random.randint(self.num_of_dungeon)
        if self.fully_informed:
            self.dung_type = (np.arange(self.num_of_dungeon) == dung_type).astype(int)
        else:
            self.dung_type = np.zeros(self.num_of_dungeon) 
        
        
        
        