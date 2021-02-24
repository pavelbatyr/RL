import argparse
import time
import collections
import datetime
import os

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

N_QUANTILES = 100  # distributional; QR-DQN

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
        self.cp_dir = cp_dir
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
            torch.save(net.state_dict(), self.cp_dir + run_name + "-best.pt")
        if mean_reward > STOP_REWARD:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_loss(batch, batch_weights, net, tgt_net, gamma, quantile_tau, device="cpu"):
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)
    batch_size = len(batch)

    states = torch.tensor(states).to(device)
    actions = torch.tensor(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.tensor(next_states).to(device)
    batch_weights = torch.tensor(batch_weights).to(device)

    qvals = net(torch.cat((states, next_states)))
    qvals_cur = qvals[:batch_size]
    qvals_next = qvals[batch_size:]

    qvals_tgt_next = tgt_net(next_states)

    max_actions_next = torch.argmax(qvals_next.mean(2), 1)
    max_actions_next_expanded = max_actions_next.view(-1, 1, 1).expand(-1, 1, N_QUANTILES)

    q_targets = qvals_tgt_next.gather(1, max_actions_next_expanded)
    q_targets[dones] = 0.0
    q_targets = q_targets * GAMMA ** N_STEPS + rewards.view(-1, 1, 1)
    assert q_targets.shape == (batch_size, 1, N_QUANTILES)

    actions_expanded = actions.view(-1, 1, 1).expand(-1, 1, N_QUANTILES)
    q_pred = qvals_cur.gather(1, actions_expanded).transpose(1, 2)
    assert q_pred.shape == (batch_size, N_QUANTILES, 1)

    td_errors = q_targets - q_pred
    assert td_errors.shape == (batch_size, N_QUANTILES, N_QUANTILES)

    K = 1
    ks = torch.full_like(td_errors, K)
    td_errors_abs = td_errors.abs()
    huber_loss = torch.where(td_errors_abs <= ks,
                             0.5 * td_errors ** 2,
                             ks * (td_errors_abs - 0.5 * ks))

    deltas = quantile_tau.view(1, 1, -1) - (td_errors.detach() < 0).float()
    quantile_loss = abs(deltas) * huber_loss / K
    assert quantile_loss.shape == (batch_size, N_QUANTILES, N_QUANTILES)

    loss = quantile_loss.sum(dim=1).mean(dim=1)
    return loss.mean(), loss + 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=float, default=STOP_REWARD,
                        help="Mean reward boundary for stop of training, default=%.2f" % STOP_REWARD)
    args = parser.parse_args()
    device = torch.device("cuda")

    cp_dir = 'checkpoints/'
    runs_dir = 'runs/'
    os.makedirs(cp_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    env = gym.make(DEFAULT_ENV_NAME)
    env = wrappers.wrap_dqn(env)

    net = dqn_model.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    date_time = datetime.datetime.now().strftime('%d-%b-%Y_%X_%f')
    run_name = f'{DEFAULT_ENV_NAME}_{date_time}'
    writer = SummaryWriter(runs_dir + run_name)

    quantile_tau = [i / N_QUANTILES for i in range(1, N_QUANTILES+1)]
    quantile_tau = torch.tensor(quantile_tau).to(device)

    agent = Agent(net, device=device)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=N_STEPS)
    buffer = experience.PrioritizedReplayBuffer(exp_source, REPLAY_SIZE, PRIO_REPLAY_ALPHA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPS)

    frame_idx = 0
    with RewardTracker(writer, net, run_name) as reward_tracker:
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
                                               gamma=GAMMA**N_STEPS, quantile_tau=quantile_tau, device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
