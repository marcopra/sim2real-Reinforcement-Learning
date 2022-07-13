"""Test an RL agent on the OpenAI Gym Hopper environment"""

import torch
import gym
import argparse
import os
from env.custom_hopper import *
from agent import Agent, Policy, StateValue
from stable_baselines3 import PPO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="model.mdl", type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=100000, type=int, help='Number of test episodes')

    return parser.parse_args()

args = parse_args()


def main():

	#env = gym.make('CustomHopper-source-v0')
	env = gym.make('CustomHopper-target-v0')

	print('Action space:', env.action_space)
	print('State space:', env.observation_space)
	print('Dynamics parameters:', env.get_parameters())
	
	
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	model = PPO.load("my_model")
	episode = 0
	episodes = 30
	returns = 0
	
	for episode in range(episodes):
		done = False
		obs = env.reset()

		test_reward = 0

		while not done:
			action, _states = model.predict(obs)
			obs, reward, done, info = env.step(action)
			env.render()

			test_reward += reward
		
		print(f"Episode: {episode} | Return: {test_reward}")
		returns+=test_reward
	
	print(f"AvgReward: {returns/episodes}")
	


	

if __name__ == '__main__':
	main()