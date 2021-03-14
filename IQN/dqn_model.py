import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dqn_pong import NUM_COSINES, K


class RainbowIQN(nn.Module):
    def __init__(self, num_channels, num_actions, embedding_dim=7*7*64):
        super().__init__()

        self.num_channels = num_channels
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

        self.dqn_net = DQNBase(num_channels, num_actions)
        self.cosine_net = CosineEmbeddingNetwork(embedding_dim=embedding_dim)
        self.quantile_net = QuantileNetwork(num_actions, embedding_dim)

    def calculate_state_embeddings(self, states):
        return self.dqn_net(states)

    def calculate_quantiles(self, taus, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        tau_embeddings = self.cosine_net(taus)
        return self.quantile_net(state_embeddings, tau_embeddings)

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None \
            else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        taus = torch.rand(
            batch_size, K, dtype=state_embeddings.dtype,
            device=state_embeddings.device)

        quantiles = self.calculate_quantiles(
            taus, state_embeddings=state_embeddings)
        assert quantiles.shape == (batch_size, K, self.num_actions)

        q = quantiles.mean(dim=1)
        assert q.shape == (batch_size, self.num_actions)

        return q

    def sample_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyFactorizedLinear):
                m.sample()


class DQNBase(nn.Module):
    def __init__(self, num_channels, n_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

    def forward(self, x):
        batch_size = x.shape[0]
        fx = x.float() / 255
        state_embedding = self.conv(fx).view(batch_size, -1)
        assert state_embedding.shape[1] == 7*7*64
        return state_embedding


class CosineEmbeddingNetwork(nn.Module):

    def __init__(self, embedding_dim, noisy_net=False):
        super().__init__()

        self.net = nn.Sequential(
            NoisyFactorizedLinear(NUM_COSINES, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = NUM_COSINES
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]

        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)

        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
            ).view(batch_size * N, self.num_cosines)

        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)

        return tau_embeddings


class QuantileNetwork(nn.Module):

    def __init__(self, num_actions, embedding_dim):
        super().__init__()

        self.advantage_net = nn.Sequential(
            NoisyFactorizedLinear(embedding_dim, 512),
            nn.ReLU(),
            NoisyFactorizedLinear(512, num_actions),
        )
        self.baseline_net = nn.Sequential(
            NoisyFactorizedLinear(embedding_dim, 512),
            nn.ReLU(),
            NoisyFactorizedLinear(512, 1),
        )

        self.num_actions = num_actions
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings, tau_embeddings):
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        assert state_embeddings.shape[1] == tau_embeddings.shape[2]

        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]

        state_embeddings = state_embeddings.view(
            batch_size, 1, self.embedding_dim)

        embeddings = (state_embeddings * tau_embeddings).view(
            batch_size * N, self.embedding_dim)

        advantages = self.advantage_net(embeddings)
        baselines = self.baseline_net(embeddings)
        quantiles = baselines + advantages - advantages.mean(1, keepdim=True)

        return quantiles.view(batch_size, N, self.num_actions)


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

    def sample(self):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()
        eps_in = self._sqrt_with_sign(self.epsilon_input.data)
        eps_out = self._sqrt_with_sign(self.epsilon_output.data)

        self.epsilon_input.copy_(eps_in)
        self.epsilon_output.copy_(eps_out)

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
