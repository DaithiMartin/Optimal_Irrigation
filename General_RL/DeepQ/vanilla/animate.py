import gym
import torch
from dqn_agent import Agent

env = gym.make('LunarLander-v2')
env.seed(0)

agent = Agent(state_size=8, action_size=4, seed=0)

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