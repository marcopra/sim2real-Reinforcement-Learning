"""Train an RL agent on the OpenAI Gym Hopper environment

TODO: implement 2.2.a and 2.2.b
"""

import torch
import gym
import argparse

from env.custom_hopper import *
from agent import Agent, Policy, StateValue

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
	observation_space_dim = env.observation_space.shape[-1]
	action_space_dim = env.action_space.shape[-1]

	policy = Policy(observation_space_dim, action_space_dim)
	Critic = StateValue(observation_space_dim)
	agent = Agent(policy, Critic, device=args.device)

	t = 1

	for episode in range(args.n_episodes):
		done = False
		train_reward = 0
		state = env.reset()  # Reset the environment and observe the initial state
		

		while not done:  # Loop until the episode is over

			t+=1

			action, action_probabilities, q_val = agent.get_action(state)
			previous_state = state

			state, reward, done, info = env.step(action.detach().cpu().numpy())

			agent.store_outcome(previous_state, state, action_probabilities, q_val, reward, done)

			train_reward += reward

			if t%51 == 0:
				t = 1
				agent.update_policy()

		

		

		# if (episode+1)%args.print_every == 0:
		if (episode+1)%100 == 0:
			print('Training episode:', episode)
			print('Episode return:', train_reward)



	torch.save(agent.policy.state_dict(), "model.mdl")
	torch.save(agent.state_value.state_dict(), "critic.mdl")

	

if __name__ == '__main__':
	main()