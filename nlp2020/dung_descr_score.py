# SCRIPT 1
# note: anytime some note is left to think about later ???? (four question marks). To check
# if all observations have been addressed just look for any remaining ????


import numpy as np

# =============================================================================
# Creating templates: basic pieces & construction
# =============================================================================

# Here we will begin creating the documents (manual and dungeon description
# we will use) DORA will use in her quest

# Element 1: dungeon type (5 dungeons)
dungeons = ['desert', 'swamp', 'mountain', 'rocky plains',  'forest']

# Element 2: monsters
monsters = ['toaxedee', 'xapossum', 'panigator', 'crocoblo', 'potsilla', 
            'keseeboon', 'zeelso', 'rhinooca', 'woosice', 'pearsoo'] 

# other monsters: 'vultopso', 'naxephant', 'grablaaps', 'dragorb', 'rooroach'

# Element 3: weapons
weapons = ['axe', 'missile', 'crossbow', 'bow', 'sword', 'hammer', 'whip']

# other: 'gun', 'dagger', 'claw', 'pistol', 'rifle', 'spear', 'bomb', 'mace', 
#           'halberds', 'scythe', 'lasso', 'stave', 'flail'


# Now we have to map monsters to dungeons
def monster2dungeon(deterministic = True, monsters = monsters):
    '''
    Mapping of monsters to dungeon
    
    deterministic: bool, opt
        Wether you want the deterministic assignment of monsters to dungeons or
        you want them to be assigned randomly (permutations of the monsters 
        vector given in input). The default is True.
        
    monsters = list, opt
        List of monsters. The default is monsters.
    Returns
    -------
    mapped_monsters : dict
        Dictionary with the final monster-dungeon mapping.
    '''
    
    if deterministic == True:
        mapped_monsters = {'desert': ['toaxedee', 'xapossum'],
                           'swamp': ['panigator', 'crocoblo'],   
                           'mountain': ['potsilla', 'keseeboon'],  
                           'rocky plains': ['zeelso', 'rhinooca'],  
                           'forest': ['woosice', 'pearsoo']}
    else:
        perm_monsters = np.random.permutation(monsters)
        mapped_monsters = {'desert': list(perm_monsters[:2]),
                           'swamp': list(perm_monsters[2:4]),   
                           'mountain': list(perm_monsters[4:6]),  
                           'rocky plains': list(perm_monsters[6:8]),  
                           'forest': list(perm_monsters[8:10])}
    

    return mapped_monsters



def weapon2monster(deterministic = True, monsters = monsters, weapons = weapons):
    '''
    
    Parameters
    ----------
    deterministic : bool, optional
        Deterministic or random assignment of weapons to monsters. The default is True.
    monsters : list, optional
        List of monsters. The default is monsters.
    weapons : list, optional
        List of weapons. The default is weapons.
    Returns
    -------
    mapped_weapons : np.array
        Matrix with n_monster number of rows and n_weapons number of columns containing the scores of each pair.
    '''
    
    if deterministic == True:
      
        mapped_weapons = [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 0.9, 0.0, 0.1, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
                          [0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0],
                          [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                          [0.0, 0.1, 0.0, 0.0, 0.9, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.9],
                          [0.0, 0.0, 0.0, 0.9, 0.0, 0.1, 0.0],
                          [0.9, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]]
        
    # ???? no else yet

    return mapped_weapons



# =============================================================================
# TEMPLATE BASE GENERATION
# =============================================================================

def dungeon_description_generator(weight_vector = None):#mapped_monsters, mapped_weapons):
    '''
    Chooses a dungeon uniformly at random and generates corresponding description
    Returns
    -------
    dungeon_description : str
        Dungeon description in string form.
    
    dugeon: str
        Sampled dungeon.
    '''
    mapped_monsters = monster2dungeon()
    mapped_weapons = weapon2monster()

    # general landscape
    feature_1 = {'desert': ['sand', 'dunes', 'cobbles', 'sand ridges', 'camels', 'coursers', 'acacias', 'spiderwebs', 'dust', 'grass'],
             'swamp': ['water', 'ponds', 'rivers', 'palafites', 'mashoofs', 'mangrove', 'fish', 'spiderwebs', 'dust', 'grass'],
             'mountain': ['snow', 'icy peaks', 'crystal flakes', 'snow flakes', 'coneflowers', 'seals', 'owls', 'spiderwebs', 'dust', 'grass'],
             'rocky plains': ['rocks', 'rocky peaks', 'oaks', 'narrow valleys', 'lichens', 'gravels', 'lynx', 'spiderwebs', 'dust', 'grass'],
             'forest': ['trees', 'shrubs', 'vines', 'lianas', 'logs', 'timber', 'wood', 'spiderwebs', 'dust', 'grass']}
    
    # nutrition of monsters therin
    feature_2 = {'desert': ['cacti', 'sandinos', 'chamyls', 'ciasacas', 'dunitos', 'bobles', 'flowerfire', 'oranges', 'strawberries', 'nuts'],
             'swamp': ['hydrophytes',  'aquarios', 'fitos', 'pandlush', 'wetios', 'marsh', 'flishyas', 'oranges', 'strawberries', 'nuts'],
             'mountain': ['sarcodes',  'blancs', 'slawer', 'iceleafs', 'iceflas', 'crystlings', 'icegtite', 'oranges', 'strawberries', 'nuts'],
             'rocky plains': ['roquetes', 'sassos', 'piedrias', 'clouds', 'sharpines', 'hardys', 'ochies', 'oranges', 'strawberries', 'nuts'],
             'forest': ['epiphytes',  'amazins', 'berries', 'moss', 'carniwer', 'amazifer', 'treflow', 'oranges', 'strawberries', 'nuts']}
    
    # temperature
    feature_3 = {'desert': ['hot', 'scolding', 'blazing', 'boiling', 'torrid', 'sizzling', 'scorching', 'strong', 'fast', 'intense'],
             'swamp': ['cold', 'freezing', 'frosty', 'icy', 'glacial', 'biting', 'polar', 'strong', 'fast', 'intense'],
             'mountain': ['cold', 'freezing', 'frosty', 'icy', 'glacial', 'biting', 'polar', 'strong', 'fast', 'intense'],
             'rocky plains': ['hot', 'scolding', 'blazing', 'boiling', 'torrid', 'sizzling', 'scorching', 'strong', 'fast', 'intense'],
             'forest': ['hot', 'scolding', 'blazing', 'boiling', 'torrid', 'sizzling', 'scorching', 'strong', 'fast', 'intense']}
    
    # weather
    feature_4 = {'desert': ['arid', 'dry', 'droughty', 'shriveled', 'barren', 'dessicated', 'impoverished', 'unstable', 'shifting', 'changing'],
             'swamp': ['humid', 'moist', 'muggy', 'oppressive', 'steamy', 'sticky', 'wet', 'unstable', 'shifting', 'changing'],
             'mountain': ['arid', 'dry', 'droughty', 'shriveled', 'barren', 'dessicated', 'impoverished', 'unstable', 'shifting', 'changing'],
             'rocky plains': ['arid', 'dry', 'droughty', 'shriveled', 'barren', 'dessicated', 'impoverished', 'unstable', 'shifting', 'changing'],
             'forest': ['humid', 'moist', 'muggy', 'oppressive', 'steamy', 'sticky', 'wet', 'unstable', 'shifting', 'changing']}
    
    # dangers (plant version)
    feature_5 = {'desert': ['spiny', 'barbed', 'thorny', 'acuminous', 'edged', 'prickly', 'bristly', 'large', 'big', 'huge'],
             'swamp': ['putrid', 'fetid', 'rancid', 'rotting', 'putrefied', 'decayed', 'moldy', 'large', 'big', 'huge'],
             'mountain': ['venomous', 'stinging', 'toxic', 'poisonous', 'virulent', 'pestiferous', 'infective', 'large', 'big', 'huge'],
             'rocky plains': ['pointy', 'sharp', 'spiked', 'spiky', 'peaked', 'sharp-cornered', 'cuspidated', 'large', 'big', 'huge'],
             'forest': ['carnivorous', 'deadly', 'vicious', 'mortal', 'lethal', 'pernicious', 'fatal', 'large', 'big', 'huge']}
    
    # other dangers
    feature_6 = {'desert': ['scorpions', 'snakes', 'intense sunlight', 'mirages', 'sandstorms', 'wasps', 'mites', 'falling', 'slipping', 'injuries'],
             'swamp': ['shifting ground', 'floods', 'leeches', 'alligators', 'crocodiles', 'piranhas', 'mosquitoes', 'falling', 'slipping', 'injuries'],
             'mountain': ['bears', 'boars', 'landslides', 'slippery slopes', 'frostbite', 'hypothermia', 'wind', 'falling', 'slipping', 'injuries'],
             'rocky plains': ['landfalls', 'sharp corners', 'sharp stones', 'slippery stones', 'black widows', 'moose', 'wolves', 'falling', 'slipping', 'injuries'],
             'forest': ['bees', 'ticks', 'lightnings', 'falling trees', 'quicksand', 'tainted water', 'ghosts', 'falling', 'slipping', 'injuries']}
    
    # random specific item to bring
    feature_7 = {'desert': ['cap', 'pair of sunglasses', 'compass', 'bottle', 'shovel', 'bucket', 'sandwich', 'book', 'notebook', 'picture'],
             'swamp': ['stick', 'boat', 'raft', 'swimsuit', 'pair of goggles', 'cup', 'towel', 'book', 'notebook', 'picture'],
             'mountain': ['scarf', 'hat', 'pair of gloves', 'windbreaker', 'pair of skis', 'backpack', 'flint rock', 'book', 'notebook', 'picture'],
             'rocky plains': ['helmet', 'binoculars', 'telescope', 'tent', 'snack', 'sleeping bag', 'headlamp', 'book', 'notebook', 'picture'],
             'forest': ['large umbrella', 'raincoat', 'vile of antidote', 'rope', 'poncho', 'bandana', 'pencil', 'book', 'notebook', 'picture']}
    
    rand_feature_1 = ['wooden', 'marble', 'metallic']
    rand_feature_2 = ['large', 'vast', 'small']
    rand_feature_3 = ['be full of perils', 'not be an easy one', 'require great courage']
    rand_feature_4 = ['horrifying', 'terrible', 'terrifying']
    rand_feature_5 = ['majestic', 'ominous', 'intimidating']
    rand_feature_6 = ['journey', 'adventure', 'quest']
    rand_feature_7 = ['frightened', 'discouraged', 'disheartened']
    rand_feature_8 = ['warrior', 'hero', 'adventurer']
    rand_feature_9 = ['wisely', 'carefully', 'attentively']
    
    if weight_vector is None:
        # Dungeon chosen sampling uniformly at random
        current_dungeon = np.random.choice(dungeons)
    else:
        current_dungeon = np.random.choice(dungeons, p = weight_vector)
    
    # Define the three sections of the story
    sect1 = ('''A small %s sign lies in front of the entrance of the dungeon. You 
    begin to read it...
    You are about to enter a %s dungeon with %s as far as the eye can see.''' 
    % (np.random.choice(rand_feature_1), 
    np.random.choice(rand_feature_2),
    np.random.choice(feature_1[current_dungeon])))
    
    sect3 = ('''No need to be %s though! You can bring weapons with you young %s. Choose among 
    the weapons next to this sign. You must make this choice %s. It’s a matter of 
    life and death! 
    Go on to your %s now and remember to bring a %s.''' 
    % (np.random.choice(rand_feature_7),
    np.random.choice(rand_feature_8),
    np.random.choice(rand_feature_9),
    np.random.choice(rand_feature_6),
    np.random.choice(feature_7[current_dungeon])))
    
    sect2 = ''
    
    rand_phrases =  ['Your path will %s: %s monsters live in the dungeon behind this %s door. ' % (np.random.choice(rand_feature_3), np.random.choice(rand_feature_4), np.random.choice(rand_feature_5)),
                    'Their nutrition consists mostly of %s and humans. ' % (np.random.choice(feature_2[current_dungeon])),
                    'Further, nature will not always be on your side: %s winds will slow down your %s. ' % (np.random.choice(feature_3[current_dungeon]), np.random.choice(rand_feature_6)),
                    '%s weather will modify the effectiveness of your weapons. ' % (np.random.choice(feature_4[current_dungeon])),
                    'Dangerously %s plants will have to be avoided. ' % (np.random.choice(feature_5[current_dungeon])),
                    'Twisted trails with the constant threat of %s will have you always one step closer to death. ' % (np.random.choice(feature_6[current_dungeon]))]    
    
    for order in np.random.permutation(range(len(rand_phrases))):
        sect2 += rand_phrases[order] 
    
     
    # Finally we put the three sections together and generate the complete description       
    dungeon_description = sect1 + '\n' + sect2 + '\n' + sect3
        
    monster = np.random.choice(mapped_monsters[current_dungeon])
    score = mapped_weapons[monsters.index(monster)]
    
    current_dungeon = (current_dungeon == np.array(dungeons)).astype(int)
    
    return dungeon_description, current_dungeon, np.array(score)


dungeon_description, _, _ = dungeon_description_generator()