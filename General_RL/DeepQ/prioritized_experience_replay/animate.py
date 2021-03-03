import gym
import torch
from Dmoney.DeepQ.prioritized_experience_replay.prioritized_dqn_agent import Agent

env = gym.make('LunarLander-v2')
env.seed(0)
class DQNConfig():
    """Configuration class for DQN"""
    def __init__(self):
        self.seed = 0

        # training
        self.n_episodes = 2000      # number of episodes
        self.max_t = 1000           # maximum frames per game
        self.update_every = 4       # number of frames before updating the network

        # environment
        self.state_size = 8
        self.action_size = 4

        # replay buffer
        self.buffer_size = int(1e5)
        self.batch_size = 64

        # annealed parameters
        self.eps_start = 1.0        # start epsilon for epsilon greedy policy
        self.eps_end = 0.01         # epsilon final value
        self.eps_decay = 0.995      # epsilon decay rate
        self.beta = 0.0

        # static parameters
        self.learning_rate = 5e-4   # learning rate
        self.gamma = 0.99           # expected future rewards discount
        self.tau = 1e-3             # fixed target weighted average parameter

config = DQNConfig()
agent = Agent(config)

state_dict = torch.load('checkpoint.pth')
agent.qnetwork_local.load_state_dict(state_dict)
agent.qnetwork_target.load_state_dict(state_dict)
eps = 0.01
for i in range(5):
    state = env.reset()
    while True:
        action = agent.act(state, eps)
        env.render()
        next_state, reward, done, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
env.close()