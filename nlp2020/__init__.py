from gym.envs.registration import register

register(
    id='nnlpDungeon-v0',
    entry_point='nlp2020.envs:nnlpDungeon',
)

register(
    id='nlpDungeon-v0',
    entry_point='nlp2020.envs:nlpDungeon',
)
