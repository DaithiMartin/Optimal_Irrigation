
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from Dmoney.Agg.Ag_agent import Agent
from Dmoney.Agg.Ag_sim import Multi_Farm_Sim
from Dmoney.Agg.config import SimConfig


def simulate(config: SimConfig):

    """
    Runs Farm Simulation with specified config setting.

    Args:
        config: configuration file

    Returns:
        list of average scores for each farmer over episodes
    """
    scores_deque = deque(maxlen=config.avg_every)
    scores = []

    for i_episode in range(1, config.num_episodes+1):
        env.reset()
        for agent in agents:
            agent.reset_noise()

        while not env.done:
            env.step(agents)

        scores_deque.append(env.scores)

        if i_episode % config.avg_every == 0:
            scores.append(np.mean(scores_deque, axis=0))

        if i_episode % config.save_every == 0:
            for i, agent in enumerate(agents):
                torch.save(agent.actor_local.state_dict(), 'checkpoints/checkpoint_actor{}.pth'.format(i))
                torch.save(agent.critic_local.state_dict(), 'checkpoints/checkpoint_critic{}.pth'.format(i))
        print("Episode: {}".format(i_episode))

    return np.array(scores)


if __name__ == '__main__':

    config = SimConfig()
    env = Multi_Farm_Sim(config)
    agents = [Agent(state_size=env.state_size, action_size=env.action_size, pre_mem_size=config.days_per_episode, random_seed=config.random_seed)
              for _ in range(config.num_farmers)]

    scores = simulate(config)

    plot_scores = [scores.T[i] for i in range(config.num_farmers)]

    for i, farmer in enumerate(plot_scores):
        plt.plot(np.arange(1, len(scores)+1), farmer, config.colors[i], label="Farmer{}".format(i+1))

    plt.title(config)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.savefig(config.plot_name(), dpi=400, bbox_inches='tight')
    plt.show()

