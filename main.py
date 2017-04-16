import sys

import gym

from deepqnetwork import DeepQNetwork
from agent import Agent
from replaymemory import ReplayMemory
from utils import Phi
import numpy as np
import random
from stats import Stats

random.seed(123)


def main(game, episodes, training_mode=False, log=False, no_ops=30):
    env = gym.make(game)
    num_actions = env.action_space.n
    dqn = DeepQNetwork(num_actions, (4, 84, 84))
    replay = ReplayMemory(100000)
    obs = env.reset()
    h, w, c = obs.shape
    phi = Phi(4, 84, 84, c, h, w)
    agent = Agent(replay, dqn, training_mode=training_mode)
    stats = Stats('results/results.csv')

    for i_episode in range(episodes):
        env.reset()

        for i in range(random.randint(1, no_ops)):
            observation, _, _, _ = env.step(0)
            pre_state = phi.add(observation)

        game_score = 0
        done = False
        t = 0

        while not done:
            t += 1
            env.render()
            action = agent.get_action(pre_state)
            observation, reward, done, _ = env.step(action)
            post_state = phi.add(observation)

            if training_mode:
                agent.update_replay_memory(pre_state, action, reward, post_state, done)
                if agent.time_step > agent.replay_start_size:
                    stats.log_time_step(agent.get_loss())

            pre_state = post_state
            game_score += reward

        print("Episode {} finished after {} time steps with score {}".format(i_episode, t, game_score))
        phi.reset()
        if agent.time_step > agent.replay_start_size:
            stats.log_game(game_score, t)

    stats.close()

    if log:
        dqn.save_model('results/model_weights.hdf5')

if __name__ == '__main__':
    game = 'Breakout-v0'
    args = sys.argv[1:]
    episodes, training_mode, log = args[:3]

    main(game, int(episodes), training_mode=bool(int(training_mode)), log=bool(int(log)))



