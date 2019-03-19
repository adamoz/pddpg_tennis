import abc
import copy
import numpy as np
from pddpg_tennis.model import Actor, Critic
from pddpg_tennis.replay_buffer import ReplayBuffer
import random
import torch
import torch.nn.functional as F
import torch.optim as optim


class AgentInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        """Save state to replay buffer and train if needed."""

    @abc.abstractmethod
    def act(self, state, add_noise=True):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            add_noise (bool): add noise to argmax over actions

        """


class Agent(AgentInterface):
    """Interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, buffer_size=int(1e6),
                 batch_size=128, tau=2e-1, weight_decay=0, gamma=0.99, lr_actor=0.0001, lr_critic=0.0003, seed=42, device='cpu'):
        """Initialize an Agent object.

        Params:
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            buffer_size (int): Replay buffer size
            batch_size (int): Size of sampled batches from replay buffer
            lr_actor (float): Learning rate
            lr_critic (float): Learning rate
            gamma (float): Reward discount
            tau (float): For soft update of target network parameters
            weight_decay (float): l2 loss during adam optimization

        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.state_size = state_size
        self.action_size = action_size
        self.weight_decay = weight_decay
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # Actor Network
        self.actor_local_1 = Actor(state_size, action_size, seed).to(device)
        self.actor_target_1 = Actor(state_size, action_size, seed).to(device)

        self.actor_local_2 = Actor(state_size, action_size, seed).to(device)
        self.actor_target_2 = Actor(state_size, action_size, seed).to(device)

        self.actor_1_optimizer = optim.Adam(list(self.actor_local_1.parameters()), lr=self.lr_actor)
        self.actor_2_optimizer = optim.Adam(list(self.actor_local_2.parameters()), lr=self.lr_actor)

        # Critic Network
        self.critic_local = Critic(state_size, action_size, seed).to(device)
        self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)

        self.noise = OUNoise((2, action_size), seed)
        self.memory = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, seed=seed, device=device)

    def __repr__(self):
        return f'Agent(state_size={self.state_size}, action_size={self.action_size}, device="{self.device}")'

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.memory.add(states, actions, rewards, next_states, dones)
        # Learn under update_rate.
        if self.memory.is_ready_to_sample():
            experiences = self.memory.sample()
            self.learn(experiences, optimizer=1)
            experiences = self.memory.sample()
            self.learn(experiences, optimizer=2)

    def act(self, states, add_noise=True):
        """Return actions for given state as per current policy.

        Params:
            state (array_like): current state
            add_noise (bool): add noise to argmax over actions
        """

        actions = np.zeros([2, self.action_size])
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local_1.eval()
        self.actor_local_2.eval()
        with torch.no_grad():
            actions[0, :] = self.actor_local_1(states[0]).cpu().data.numpy()
            actions[1, :] = self.actor_local_2(states[1]).cpu().data.numpy()
        self.actor_local_1.train()
        self.actor_local_2.train()

        if add_noise:
            actions += self.noise.sample()
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma=None, tau=None, optimizer=1):
        """Update value parameters using given batch of experience tuples.

        Params:
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done, ss, aa, s's') tuples
            gamma (float): discount factor
            tau (float): For soft update of target network parameters
        """

        if gamma is None:
            gamma = self.gamma

        if tau is None:
            tau = self.tau

        states, actions, rewards, next_states, dones, other_states, other_actions, other_next_states = experiences

        # ------------------ concat all values for shared critic ----------------- #
        all_states = torch.cat((states, other_states), dim=1).to(self.device)
        all_actions = torch.cat((actions, other_actions), dim=1).to(self.device)
        all_next_states = torch.cat((next_states, other_next_states), dim=1).to(self.device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_actions = self.actor_target_1(next_states)
        other_next_actions = self.actor_target_2(other_next_states)
        all_next_actions = torch.cat((next_actions, other_next_actions), dim=1).to(self.device)

        Q_targets_next = self.critic_target(all_next_states, all_next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(all_states, all_actions)

        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local_1(states)
        other_actions_pred = self.actor_local_2(other_states)
        all_actions_pred = torch.cat((actions_pred, other_actions_pred), dim=1).to(self.device)

        actor_loss = -self.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        if optimizer == 1:
            self.actor_1_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_1_optimizer.step()
            self.soft_update(self.actor_local_1, self.actor_target_1, tau)
        else:
            self.actor_2_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_2_optimizer.step()
            self.soft_update(self.actor_local_2, self.actor_target_2, tau)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, file_name):
        torch.save(self.actor_local_1.state_dict(), file_name + '_1_actor.pth')
        torch.save(self.actor_local_2.state_dict(), file_name + '_2_actor.pth')
        torch.save(self.critic_local.state_dict(), file_name + '_critic.pth')

    def load(self, file_name):
        self.actor_local_1.load_state_dict(torch.load(file_name + '_1_actor.pth'))
        self.actor_target_1.load_state_dict(torch.load(file_name + '_1_actor.pth'))
        self.actor_local_2.load_state_dict(torch.load(file_name + '_2_actor.pth'))
        self.actor_target_2.load_state_dict(torch.load(file_name + '_2_actor.pth'))
        self.critic_local.load_state_dict(torch.load(file_name + '_critic.pth'))
        self.critic_target.load_state_dict(torch.load(file_name + '_critic.pth'))


class OUNoise:
    """Ornstein-Uhlenbeck process"""

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.05, sigma_decay=1.):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state
