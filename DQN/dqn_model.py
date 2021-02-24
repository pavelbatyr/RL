import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dqn_pong import N_QUANTILES


class NoisyFactorizedLinear(nn.Linear):
    """
    NoisyNet layer with factorized gaussian noise
    N.B. nn.Linear already initializes weight and bias to
    """
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias=bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

    def _sqrt_with_sign(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x))

    def forward(self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        eps_in = self._sqrt_with_sign(self.epsilon_input.data)
        eps_out = self._sqrt_with_sign(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()
        noise_v = torch.mul(eps_in, eps_out)
        return F.linear(input, self.weight + self.sigma_weight * noise_v, bias)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._conv_out_size(input_shape)
        
        # saparate branches for state value and advantages of actions
        self.fc_val = nn.Sequential(
            NoisyFactorizedLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyFactorizedLinear(512, N_QUANTILES)
        )
        self.fc_adv = nn.Sequential(
            NoisyFactorizedLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyFactorizedLinear(512, n_actions * N_QUANTILES)
        )

    def _conv_out_size(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.shape[0]
        fx = x.float() / 256
        conv_out = self.conv(fx).view(batch_size, -1)
        val = self.fc_val(conv_out).view(batch_size, 1, N_QUANTILES)
        adv = self.fc_adv(conv_out).view(batch_size, -1, N_QUANTILES)
        adv_mean = adv.mean(dim=1, keepdim=True)
        return val + adv - adv_mean
