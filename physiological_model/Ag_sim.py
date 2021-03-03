import numpy as np
import matplotlib.pyplot as plt
from Dmoney.Agg.config import SimConfig

class Multi_Farm_Sim:
    def __init__(self, config: SimConfig):
        # environment variables
        self.state_size = 1 + 3 * config.num_crops     # [available_water
        self.action_size = config.num_crops

        # episode variables
        self.total_days = config.days_per_episode
        self.max_water = config.max_water
        self.min_water = config.min_water
        self.water_source = config.source_water_function
        self.num_farmers = config.num_farmers
        self.num_crops = config.num_crops
        self.source_water = self.water_source
        self.original_crop_prices = config.crop_prices
        self.random_seed = np.random.seed(config.random_seed)

        # instance variables
        self.day = 0
        self.done = False
        self.agent_priority = config.agent_priority
        self.scores = []

        # State variables
        self.available_water = self.source_available_water(self.day)
        self.crop_water = [[0 for i in range(self.num_crops)] for i in range(self.num_farmers)]
        self.crop_growth = [[0 for i in range(self.num_crops)] for i in range(self.num_farmers)]
        self.crop_prices = self.original_crop_prices


    def reset(self):
        """
        Resets environment to original settings.

        Returns:
            None
        """
        # instance variables
        self.day = 0
        self.done = False
        self.scores = []

        # State variables
        self.available_water = self.source_available_water(self.day)
        self.crop_water = [[0 for i in range(self.num_crops)] for i in range(self.num_farmers)]
        self.crop_growth = [[0 for i in range(self.num_crops)] for i in range(self.num_farmers)]
        self.crop_prices = self.original_crop_prices

        return None

    def source_available_water(self, day):
        """Gets available water for each farmer"""
        self.available_water = [self.source_water[day] for i in range(self.num_farmers)]

        return None

    def update_available_water(self, priority_index, action):
        """
        Updates all available water down stream from agent indicated by priority index.

        Args:
            priority_index: indicates at what point in the river continuum to update flows
            action: amount of water removed by agent
        """
        total_removed = np.sum(action)
        for i in range(priority_index, len(self.available_water)):

            self.available_water[i] = max(self.available_water[i] - total_removed, 1e-3)

        return None


    def update_crop_water(self, priority_index, action):
        """
        Updates water levels in each crop with the action taken by the agent.
        An additional random amount is removed for growth/transpiration.

        Args:
            priority_index: indicates which farmers crops are being updated
            action: how much water for each crop was removed from river

        Returns:
            None

        """
        for i, crop in enumerate(self.crop_water[priority_index]):
            self.crop_water[priority_index][i] = crop + action[i] - np.abs(np.random.randn())

        return None

    def update_crop_growth(self, priority_index, action):
        """
        Updates crop growth for the respective farmer logrithmically proportional to water added.
        Args:
            priority_index: idicates which farmers corps are updated
            action: how much water for each crop

        Returns:
            None

        """
        for i, crop in enumerate(self.crop_growth[priority_index]):
            if action[i] < 0:
                print('invalid action')
            self.crop_growth[priority_index][i] = crop + np.abs((np.log(action[i] + 1e-6) + (np.random.randn() / 5)))

            return None

    def update_crop_price(self):
        """Updates crop prices with random market fluctuations and current expected supply"""


    def step(self, farmers):
        """
        Takes a step in the environment and updates all relevant instance attributes.
        Args:
            farmers: vector of Agent farmers

        Returns:
            True if season is done else False
        """

        if self.day < self.total_days:
            """
            Indicates non terminal state, update everything but no rewards
            """
            self.source_available_water(self.day)
            for priority_index in self.agent_priority:
                state = np.array(
                                [self.available_water[priority_index]] +
                                self.crop_water[priority_index] +
                                self.crop_growth[priority_index] +
                                self.crop_prices)

                action = list(farmers[priority_index].act(state))
                self.update_available_water(priority_index, action)
                self.update_crop_water(priority_index, action)
                self.update_crop_growth(priority_index, action)
                reward = [0]
                next_state = np.array(
                                [self.available_water[priority_index]] +
                                self.crop_water[priority_index] +
                                self.crop_growth[priority_index] +
                                self.crop_prices)
                farmers[priority_index].step(state, action, reward, next_state, self.done)
                self.update_crop_price()

            self.done = False


        else:
            """
            Indicates terminal state, farmer agents sell their crops and rewards are calculated
            """
            self.source_available_water(self.day)
            for priority_index in self.agent_priority:
                state = np.array(
                                [self.available_water[priority_index]] +
                                self.crop_water[priority_index] +
                                self.crop_growth[priority_index] +
                                self.crop_prices)
                action = farmers[priority_index].act(state)
                self.update_available_water(priority_index, action)
                self.update_crop_water(priority_index, action)
                self.update_crop_growth(priority_index, action)
                reward = np.sum(np.array(self.crop_growth[priority_index]) * self.crop_prices)
                next_state = np.array(
                                [self.available_water[priority_index]] +
                                self.crop_water[priority_index] +
                                self.crop_growth[priority_index] +
                                self.crop_prices)
                farmers[priority_index].step(state, action, reward, next_state, self.done)
                farmers[priority_index].update_memory(reward)
                self.scores.append(np.sum(reward))

                self.done = True

        self.day += 1
