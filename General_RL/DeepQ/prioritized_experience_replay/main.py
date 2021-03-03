import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import csv

from Dmoney.DeepQ.prioritized_experience_replay.prioritized_dqn_agent import Agent

env = gym.make('LunarLander-v2')
env.seed(0)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

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
        

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Train DeepQ Learning Agent.

    Args:
        n_episodes: (int) maximum number of training episodes
        max_t: (int) maximum number of timesteps per episode
        eps_start: (float) starting value of epsilon, for epsilon-greedy action selection
        eps_end: (float) minimum value of epsilon
        eps_decay: (float) multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores


config = DQNConfig()

agent = Agent(config)

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.title("Prioritized Replay")
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


with open("scores.csv", 'w') as outfile:
    write = csv.writer(outfile)
    write.writerow(scores)


