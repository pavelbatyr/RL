import argparse
import time
import collections
import datetime
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import gym

import wrappers
import dqn_model
from agent import Agent
import experience
import utils


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
STOP_REWARD = 19.5

N_STEPS = 2  # number of steps to unroll Bellman

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100_000

# IQN params:
N = 64
N_dash = 64
K = 32
NUM_COSINES = 64
KAPPA = 1.0  # for Huber loss

GAMMA = 0.99  # discounting factor
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
ADAM_EPS = 0.01 / 32
REPLAY_SIZE = 100_000
REPLAY_START_SIZE = 10000
SYNC_TARGET_FRAMES = 1000


class RewardTracker:
    def __init__(self, writer, net, run_name, cp_dir):
        self.writer = writer
        self.save_path = str(cp_dir / f'{run_name}-best.pt')
        self.best_reward = None

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s" % (
            frame, len(self.total_rewards), mean_reward, speed), flush=True)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if self.best_reward is None:
            self.best_reward = mean_reward
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            torch.save(net.state_dict(), self.save_path)
        if mean_reward > STOP_REWARD:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)
    batch_size = len(batch)

    states = torch.tensor(states, device=device)
    actions = torch.tensor(actions, device=device)
    dones = torch.tensor(dones, device=device)
    rewards = torch.tensor(rewards, device=device)
    next_states = torch.tensor(next_states, device=device)
    batch_weights = torch.tensor(batch_weights, device=device)

    state_embeddings = net.calculate_state_embeddings(states)

    taus = torch.rand(
        batch_size, N, dtype=state_embeddings.dtype,
        device=device)

    current_sa_quantiles = utils.evaluate_quantile_at_action(
        net.calculate_quantiles(taus, state_embeddings=state_embeddings),
        actions.unsqueeze(-1))
    assert current_sa_quantiles.shape == (batch_size, N, 1)

    with torch.no_grad():
        net.sample_noise()
        next_q = net.calculate_q(states=next_states)
        next_actions = torch.argmax(next_q, dim=1, keepdim=True)
        assert next_actions.shape == (batch_size, 1)

        next_state_embeddings = tgt_net.calculate_state_embeddings(next_states)

        tau_dashes = torch.rand(
            batch_size, N_dash, dtype=state_embeddings.dtype,
            device=device)

        next_sa_quantiles = utils.evaluate_quantile_at_action(
            tgt_net.calculate_quantiles(tau_dashes, state_embeddings=next_state_embeddings), 
            next_actions).transpose(1, 2)
        assert next_sa_quantiles.shape == (batch_size, 1, N_dash)

        target_sa_quantiles = rewards.view(-1, 1, 1) + (
            1.0 - dones.view(-1, 1, 1)) * gamma * next_sa_quantiles
        assert target_sa_quantiles.shape == (batch_size, 1, N_dash)

    td_errors = target_sa_quantiles - current_sa_quantiles
    assert td_errors.shape == (batch_size, N, N_dash)

    loss, losses = utils.calculate_quantile_huber_loss(td_errors, taus, KAPPA)
    # TODO td_errors-based priority?
    return loss, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=float, default=STOP_REWARD,
                        help="Mean reward boundary for stop of training, default=%.2f" % STOP_REWARD)
    args = parser.parse_args()
    device = torch.device("cuda")

    project_dir = Path(__file__).resolve().parent
    runs_dir = project_dir / 'runs'
    cp_dir = project_dir / 'checkpoints'
    os.makedirs(str(runs_dir), exist_ok=True)
    os.makedirs(str(cp_dir), exist_ok=True)

    date_time = datetime.datetime.now().strftime('%d-%b-%Y_%X_%f')
    run_name = f'{DEFAULT_ENV_NAME}_{date_time}'
    run_path = str(runs_dir / run_name)
    writer = SummaryWriter(run_path)
    
    env = gym.make(DEFAULT_ENV_NAME)
    env = wrappers.wrap_dqn(env)

    net = dqn_model.RainbowIQN(num_channels=env.observation_space.shape[0],
                               num_actions=env.action_space.n
                               ).to(device)
    tgt_net = dqn_model.RainbowIQN(num_channels=env.observation_space.shape[0],
                                   num_actions=env.action_space.n
                                   ).to(device)

    agent = Agent(net, device=device)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=N_STEPS)
    buffer = experience.PrioritizedReplayBuffer(exp_source, REPLAY_SIZE, PRIO_REPLAY_ALPHA)  # TODO tree implementation

    # TODO eps schedule
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

    frame_idx = 0
    with RewardTracker(writer, net, run_name, cp_dir) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(BATCH_SIZE, beta)
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net,
                                               gamma=GAMMA**N_STEPS, device=device)

            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
