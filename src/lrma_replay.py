import random
import torch
import numpy as np


class LRMAPersistentReplayBuffer:
    """
    Persistent Experience Replay Buffer for LRMA CTDE Training (IEEE TNSE 2025 Paper Algorithm 1, lines 2, 22-25).
    Transitions persist across gradient updates. Sampling retrieves random minibatches of size B (default 64).
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done=False):
        """
        Appends experience transition (state, action, reward, next_state, done) to persistent buffer.
        """
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size=64):
        """
        Samples a random minibatch of transitions from the persistent buffer.
        Returns numpy arrays / lists for (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )

    def __len__(self):
        return len(self.buffer)
