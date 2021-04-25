import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor (Deterministic Policy!) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """
        Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc_units (int): Number of nodes in first hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed) if seed is not None else None

        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)

        self.fc_water_head_1 = nn.Linear(fc_units, fc_units)
        self.fc_water_head_2 = nn.Linear(fc_units, 101)

        self.fc_land_head_1 = nn.Linear(fc_units, fc_units)
        self.fc_land_head_2 = nn.Linear(fc_units, 101)

        self.dropout_1 = nn.Dropout(p=0.25)
        self.dropout_2 = nn.Dropout(p=0.25)
        self.dropout_3 = nn.Dropout(p=0.25)
        self.dropout_4 = nn.Dropout(p=0.25)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

        self.fc_water_head_1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_water_head_2.weight.data.uniform_(-3e-3, 3e-3)

        self.fc_land_head_1.weight.data.uniform_(-3e-3, 3e-3)
        self.fc_land_head_2.weight.data.uniform_(-3e-3, 3e-3)

        return None

    def forward(self, state):
        """
        Policy network that maps states -> action values! not actions

        Args:
            state vector (torch.tensor):
            [batch, available_water, available_land, crops_encoding, cost_encoding]
        """

        x1 = self.dropout_1(F.relu(self.fc1(state)))
        x2 = self.dropout_2(F.relu(self.fc2(x1)))

        x_water = self.dropout_3(F.relu(self.fc_water_head_1(x2)))
        x_land = self.dropout_4(F.relu(self.fc_land_head_1(x2)))

        water_values = self.fc_water_head_2(x_water)
        land_values = self.fc_land_head_2(x_land)

        return torch.stack((water_values, land_values))


class Critic(nn.Module):
    """Critic (Action Value!) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
        """Initialize parameters and build model.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed) if seed is not None else None
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)

    def forward(self, state, action):
        """
        Critic (value) network that maps (state, action) pairs -> Q-values.

        Args:
            state (Vector, torch.tensor): [available_water, CropWater_n, CropGrowth_n, CropPrices_n]
                                          Where n is the number of crops dependant of simulation
            action (Vector, torch.tensor): [WaterAmount_n] Where n is the number of crops.
        """
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)
