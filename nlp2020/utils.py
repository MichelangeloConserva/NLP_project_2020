import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

def smooth(array, smoothing_horizon=100., initial_value=0.):
  """Smoothing function for plotting. Credit to Deep Mind Lectures at UCL"""
  smoothed_array = []
  value = initial_value
  b = 1./smoothing_horizon
  m = 1.
  for x in array:
    m *= 1. - b
    lr = b/(1 - m)
    value += lr*(x - value)
    smoothed_array.append(value)
  return np.array(smoothed_array)


def multi_bar_plot(algs, n_mission_per_episode, test_trials, n_test_trials):
    # Multi bars plot
    spacing = np.linspace(-1,1, len(algs))
    width = spacing[1] - spacing[0]
    missions = np.arange((n_mission_per_episode+1)*4, step = 4)
    for (i,(_,(agent,_,_,_,_,col,_))) in enumerate(algs.items()):
        c = Counter(test_trials[agent.name])
        counts = np.zeros(n_mission_per_episode+1)
        for k,v in c.items(): counts[k] = v/n_test_trials

        assert round(sum(counts),5) == 1
        plt.bar(missions + spacing[i], 
                counts, width, label = agent.name, color = col, edgecolor="black")
        
    plt.xlabel("Consecutive mission, i.e. length of the episode")
    plt.xticks(missions,range(n_mission_per_episode+1))
    plt.legend()