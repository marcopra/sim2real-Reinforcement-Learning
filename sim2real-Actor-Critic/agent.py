import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TODO 2.2.b: critic network for actor-critic algorithm


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

    
        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


       

        
        return normal_dist

class StateValue(torch.nn.Module):
    def __init__(self, state_space, reward_dim = 1):
        super().__init__()
        self.state_space = state_space
        self.reward_dim = reward_dim
        self.hidden = 32
        self.tanh = torch.nn.Tanh()


        """
            Critic network
        """
        # TODO 2.2.b: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_rew = torch.nn.Linear(self.hidden, reward_dim)


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        
    
        """
            Critic
        """
        # TODO 2.2.b: forward in the critic network
        x_actor = self.tanh(self.fc1_critic(x))
        x_actor = self.tanh(self.fc2_critic(x_actor))
        predicted_rew = self.fc3_critic_rew(x_actor)


        
        return predicted_rew


class Agent(object):
    def __init__(self, policy, state_value, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        self.state_value = state_value.to(self.train_device)
        self.optimizerAction = torch.optim.Adam(policy.parameters(), lr=1e-3)
        self.optimizerCritic = torch.optim.Adam(state_value.parameters(), lr=1e-3)

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.discountedRewards = []
        self.q_value = []
        self.next_q_value = []

        

    def update_policy(self):
       
                    
        lossActor = 0
       

        #
        # TODO 2.2.b:
        #             - compute boostrapped discounted return estimates
        #             - compute advantage terms
        #             - compute actor loss and critic loss
        #             - compute gradients and step the optimizer
        #

        lossCritic = 0
        lossActor = 0
        if(len(self.rewards) != len(self.q_value)):
            print("Errore nei parametri rewards e q_value")
        else:
            
            for lp,q_t, q_t_1, r_t, done in zip(self.action_log_probs,self.q_value, self.next_q_value, self.rewards, self.done):
                if done:
                    lossCritic += ((r_t) - q_t)**2
                    lossActor += (r_t  - q_t)*lp
                else:
                    lossCritic += ((r_t + self.gamma*q_t_1.detach()) - q_t)**2
                    lossActor += (r_t + self.gamma*q_t_1 - q_t)*lp
            
            
        lossCritic = lossCritic/len(self.rewards)
        lossActor = -lossActor/len(self.rewards)
        #Pytorch keep tracks of the losses' history and it knows where to backpropagate (?????)
        losses = lossActor + lossCritic     

        self.optimizerAction.zero_grad()
        self.optimizerCritic.zero_grad()
        losses.backward()
        
        self.optimizerCritic.step()
        self.optimizerAction.step()


        self.reset_all()
        return        

    def get_action(self, state, evaluation=False):
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            q_value = self.state_value(x)

            return action, action_log_prob, q_value

    def store_outcome(self, state, next_state, action_log_prob, q_value, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(reward)
        self.done.append(done)
        self.q_value.append(q_value)
        self.next_q_value.append(self.state_value(torch.from_numpy(next_state).float().to(self.train_device)))

        self.discountedRewards = []
        for t in range(len(self.rewards)):
            G = 0.0
            for k, r in enumerate(self.rewards[t:]):
                G += (self.gamma**k)*r
            
            self.discountedRewards.append(G)
        

    

    def reset_all(self):
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []
        self.q_value = []
        self.discountedRewards = []
        self.next_q_value = []

