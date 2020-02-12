from gym.envs.registration import register

register(
    id='BaseDungeon',
    entry_point='nlp2020.envs:BaseDungeon',
)

register(
    id='nnlpDungeon',
    entry_point='nlp2020.envs:nnlpDungeon',
)
