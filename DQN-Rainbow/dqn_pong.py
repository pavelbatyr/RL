import argparse
import time
import collections
import datetime

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

REWARD_STEPS_DEFAULT = 2  # number of steps to unroll Bellman

PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100_000

GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100_000
REPLAY_START_SIZE = 10000
SYNC_TARGET_FRAMES = 1000

Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RewardTracker:
    def __init__(self, writer, net, run_name):
        self.writer = writer
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
            torch.save(net.state_dict(), run_name + "-best.pt")
        if mean_reward > STOP_REWARD:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = utils.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # next state distribution
    # dueling arch -- actions from main net, distr from tgt_net

    # calc at once both next and cur states
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    next_actions_v = next_qvals_v.max(1)[1]
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_softmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()
 
    dones = dones.astype(np.bool)

    # project our distribution using Bellman update
    proj_distr = utils.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # calculate net output
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights_v * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=float, default=STOP_REWARD,
                        help="Mean reward boundary for stop of training, default=%.2f" % STOP_REWARD)
    args = parser.parse_args()
    device = torch.device("cuda")

    env = gym.make(DEFAULT_ENV_NAME)
    env = wrappers.wrap_dqn(env)

    net = dqn_model.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)

    date_time = datetime.datetime.now().strftime('%d-%b-%Y_%X_%f')
    run_name = f'{DEFAULT_ENV_NAME}_{date_time}'
    writer = SummaryWriter('runs/' + run_name)

    agent = Agent(lambda x: net.qvals(x), device=device)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS_DEFAULT)
    buffer = experience.PrioritizedReplayBuffer(exp_source, REPLAY_SIZE, PRIO_REPLAY_ALPHA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

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
                                               gamma=GAMMA**REWARD_STEPS_DEFAULT, device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
