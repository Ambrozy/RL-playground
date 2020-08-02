import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


class PlotDurations:
    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return np.concatenate([
            ret[:n-1] / np.arange(1, n),
            ret[n-1:] / n,
        ])

    @staticmethod
    def __plot_durations(episode_durations, title='Training...'):
        plt.figure(2)
        plt.clf()
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(episode_durations, label='Episode duration')
        # Take 100 episode averages and plot them too
        if len(episode_durations) >= 2:
            n = np.min([len(episode_durations), 100])
            means = PlotDurations.moving_average(episode_durations, n=n)
            plt.plot(means, label='Mean episode duration')

        plt.legend(loc='upper left')
        plt.pause(0.1)  # pause a bit so that plots are updated
        if is_ipython:
            display.clear_output(wait=True)

    @staticmethod
    def train_plot(episode_durations):
        PlotDurations.__plot_durations(episode_durations, 'Training...')
        plt.show()

    @staticmethod
    def complete_plot(episode_durations):
        PlotDurations.__plot_durations(episode_durations, 'Complete')
        plt.show()
