import abc
import numpy as np
from unityagents import UnityEnvironment


class EnvInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self):
        """Reset environment and return starting state."""

    @abc.abstractmethod
    def step(self, actions):
        """Do next step and return (states, rewards, dones info."""

    @abc.abstractmethod
    def close(self):
        """Close environment."""


class UnityEnvironmentWrapper(EnvInterface):
    def __init__(self, env_binary='../bin/tennis/Tennis.x86_64', train_mode=True):
        self.env = UnityEnvironment(file_name=env_binary)

        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]
        self.train_mode = train_mode
        self.info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        self.num_agents = len(self.info.agents)

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        states = env_info.vector_observations
        return states

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return (next_states, rewards, np.array(dones) * 1)

    def close(self):
        self.env.close()

    @property
    def state_size(self):
        state = self.info.vector_observations[0]
        return len(state)

    @property
    def action_size(self):
        action_size = self.brain.vector_action_space_size
        return action_size
