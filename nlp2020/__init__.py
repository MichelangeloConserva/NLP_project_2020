from gym.envs.registration import register

register(
    id='BaseDungeon-v0',
    entry_point='nlp2020.envs:BaseDungeon',
)

register(
    id='nnlpDungeon-v0',
    entry_point='nlp2020.envs:nnlpDungeon',
)
