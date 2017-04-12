import sys

import gym

from deepqnetwork import DeepQNetwork
from agent import Agent
from replaymemory import ReplayMemory
from utils import Phi
import numpy as np

import time


def main(game, episodes, game_time_limit, training_mode=False, log=False):
    env = gym.make(game)
    num_actions = env.action_space.n
    dqn = DeepQNetwork(num_actions, (4, 84, 84))
    replay = ReplayMemory(100000)
    obs = env.reset()
    h, w, c = obs.shape
    phi = Phi(4, 84, 84, c, h, w)
    agent = Agent(replay, dqn, training_mode=training_mode)

    for i_episode in range(episodes):
        observation = env.reset()
        pre_state = phi.add(observation)
        game_score = 0
        for t in range(game_time_limit):
            env.render()
            action = agent.get_action(pre_state)
            observation, reward, done, info = env.step(action)
            post_state = phi.add(observation)
            agent.update_replay_memory(pre_state, action, reward, post_state, done)
            pre_state = post_state
            game_score += reward

            if done:
                print("Episode finished after {} timesteps with score {}".format(t+1, game_score))
                break
        phi.reset()
    if log:
        dqn.save_model('results/model_weights.hdf5')

if __name__ == '__main__':
    game = 'Breakout-v0'
    args = sys.argv[1:]
    episodes, game_time_limit, training_mode, log = args[:4]

    main(game, int(episodes), int(game_time_limit), training_mode=bool(training_mode), log=bool(log))



