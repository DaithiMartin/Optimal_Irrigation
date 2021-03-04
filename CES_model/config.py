
import numpy as np

class SimConfig:
    # TODO: determine if i even need a config class. perhaps start without a config class then modularize as needed
    def __init__(self):

        self.num_crops = 2

        # TODO: use a dictionary to hold these values for different watersheds, then access watershed values via init argument
        # hydrology function, current estimate based on lower Clark Fork
        self.water_mu = 1.2e6       # cfs
        self.water_sigma = 1e5      # cfs
        self.water_dist = self.water_sigma * np.random.randn(100) + self.water_mu

        # simulation parameters
        self.number_farmers = 3
        self.farmer_priority = [0, 1, 2]
        self.random_seed = 0



        # agent parameters
        # [available_water, available_land, crop_identity_vec, crop_price_vec]
        # FIXME: THIS WILL NEED TO CHANGE WHEN self.crop_list CHANGES TO ONE HOT VECTOR
        self.state_size = 2 + 2 * self.num_crops
        self.action_size = 2 * self.num_crops
        self.memory_size = 10

