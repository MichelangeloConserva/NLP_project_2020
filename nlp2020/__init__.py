from gym.envs.registration import register

try:
    register(
        id='nnlpDungeon-v0',
        entry_point='nlp2020.envs:nnlpDungeon',
    )
except: pass

try:
    register(
        id='nlpDungeon-v0',
        entry_point='nlp2020.envs:nlpDungeon',
    )
except: pass
