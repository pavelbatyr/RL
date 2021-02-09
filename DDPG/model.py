import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 400), nn.ReLU(),
            nn.Linear(400, 300), nn.ReLU(),
            nn.Linear(300, act_size), nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
    

class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()
        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )
        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300), nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
