"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse

from env.custom_hopper import *
from agent import Agent, Policy, StateValue
from stable_baselines3 import PPO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=20000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()

args = parse_args()

def main():

	env = gym.make('CustomHopper-source-v0')
	# env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())


	"""
		Training
	"""

	model = PPO("MlpPolicy", env, verbose=1)
	model.learn(total_timesteps=1000000)

	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	



	model.save("my_model")
	

	

if __name__ == '__main__':
	main()