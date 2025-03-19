import os
from tqdm import tqdm
import torch
import wandb
import random
import torch
import numpy as np
import pickle

class ReplayBuffer:
    def __init__(self, max_buffer_size = 1000000, max_ep_len = 1000):
        self.buffer = []
        self.max_buffer_size = max_buffer_size
        self.max_ep_len = max_ep_len     
    
    def add_episodes(self, new_traj):
        """
        :param train_buffer: key: prefix_length, value: list of trajectories, may be empty
        :param buffer_size: size of the buffer for each prefix_length
        """
        self.buffer.extend(new_traj)
        if len(self.buffer) > self.max_buffer_size:
            random.shuffle(self.buffer)
            self.buffer = self.buffer[(self.max_buffer_size//5):]

    def get_loader(self, batch_size=2048):
        # dset = data.dataset(self.buffer)
        return torch.utils.data.DataLoader(self.buffer, batch_size, shuffle=True, drop_last=False)
    
    