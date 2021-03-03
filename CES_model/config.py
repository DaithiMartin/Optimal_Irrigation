import numpy as np
from datetime import datetime


class SimConfig:
    """
    Defines Configuration for Ag Simulator
    """

    def __init__(self):
        # Training parameters
        self.num_episodes = 1000
        self.avg_every = 10
        self.save_every = 100
        self.true_random = False
        self.random_seed = None if self.true_random else 10

        # Episode parameters
        self.days_per_episode = 20  # growing season
        self.num_crops = 2  # assumed equal distribution over land
        self.num_farmers = 3
        self.max_water = 1000  # discharge volume/time
        self.min_water = 50  # discharge volume/time
        self.crop_prices = [100, 50]

        # order of agent priority
        self.reversed_agent_priority = False
        self.agent_priority = np.flip(
            np.arange(self.num_farmers)).tolist() if self.reversed_agent_priority else np.arange(
            self.num_farmers).tolist()

        # Crop and water functions
        self.water_mu = 100
        self.water_sigma = 20
        self.source_water_function = self.water_sigma * np.random.randn(self.days_per_episode + 1) + self.water_mu

        # Agent parameters
        self.agent_type = "A2C"

        # plotting parameters
        self.colors = ['r-', 'b-', 'g-', 'c-', 'm-', 'y-', 'k-']

    def __str__(self):
        out_string = "Num Episodes: {:<6}   Days/episode: {}\n" \
                     "Num Crops: {:<6}      Num Farmers: {}\n" \
                     "Random: {}         Agent priority: {}"

        return out_string.format(self.num_episodes, self.days_per_episode, self.num_crops, self.num_farmers,
                                 self.true_random, self.agent_priority)

    def plot_name(self):
        return "plots/episodes_{}_random_{}_order_{}_agents_{}_{}.png".format(self.num_episodes,
                                                                                   self.true_random,
                                                                                   self.agent_priority,
                                                                                   self.num_farmers,
                                                                                   datetime.now()
                                                                                   )





# import matplotlib.pyplot as plt
# x = np.arange(1000)
# config = SimConfig
# plt.plot(x, )