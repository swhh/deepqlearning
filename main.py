import gym
import cProfile

from deepqnetwork import DeepQNetwork
from agent import Agent
from replaymemory import ReplayMemory

from utils import Phi
import numpy as np


env = gym.make('Breakout-v0')
num_actions = env.action_space.n

dqn = DeepQNetwork(num_actions, (4, 84, 84))

replay = ReplayMemory(100000)

phi = Phi(84, 84, 4)

agent = Agent(replay, dqn, phi)

for i_episode in range(200):
    observation = env.reset()
    for t in range(10):
        env.render()
        pre_state = observation
        action = agent.get_action(observation)
        observation, reward, done, info = env.step(action)
        agent.update_replay_memory(pre_state, action, reward, observation, done)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


