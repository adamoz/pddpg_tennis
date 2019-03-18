from collections import namedtuple, deque
import numpy as np
from numpy.random import choice
import random
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size=int(1e5), batch_size=64, seed=0, device='cpu'):
        """Initialize a ReplayBuffer object.

        Params:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            device (str): device where tensors are proecssed
        """

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done",
                                                                "other_state", "other_action", "other_next_state"])
        self.device = device
        self.seed = random.seed(seed)

    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        e1 = self.experience(states[0], actions[0], rewards[0], next_states[0], dones[0], states[1], actions[1], next_states[1])
        e2 = self.experience(states[1], actions[1], rewards[1], next_states[1], dones[1], states[0], actions[0], next_states[0])
        self.memory.append(e1)
        self.memory.append(e2)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        other_states = torch.from_numpy(np.vstack([e.other_state for e in experiences if e is not None])).float().to(self.device)
        other_actions = torch.from_numpy(np.vstack([e.other_action for e in experiences if e is not None])).float().to(self.device)
        other_next_states = torch.from_numpy(np.vstack([e.other_next_state for e in experiences if e is not None])).float().to(self.device)

        return (states, actions, rewards, next_states, dones, other_states, other_actions, other_next_states)

    def is_ready_to_sample(self):
        return len(self) > self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
