import argparse
import time
import numpy as np
import collections
import datetime

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


DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
STOP_REWARD = 19.5

REWARD_STEPS_DEFAULT = 2  # number of steps to unroll Bellman
DOUBLE_DQN = True
GAMMA = 0.99
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
REPLAY_SIZE = 100_000
REPLAY_START_SIZE = 10000
SYNC_TARGET_FRAMES = 1000


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
        if frame % 500 == 0:
            for layer_idx, sigma_l2 in enumerate(net.noisy_layers_sigma_snr()):
                writer.add_scalar(f"sigma_snr_layer_{layer_idx+1}", sigma_l2, frame)
        if self.best_reward is None:
            self.best_reward = mean_reward
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            torch.save(net.state_dict(), run_name + "-best.pt")
        if mean_reward > STOP_REWARD:
            print("Solved in %d frames!" % frame)
            return True
        return False


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=bool), np.array(last_states, copy=False)


def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu", double=True):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states, dtype=torch.float).to(device)
    next_states_v = torch.tensor(next_states, dtype=torch.float).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    if double:
        next_state_actions = net(next_states_v).max(1)[1]
        next_state_values = tgt_net(next_states_v).gather(1, next_state_actions.unsqueeze(-1)).squeeze(1)
    else:
        next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward", type=float, default=STOP_REWARD,
                        help="Mean reward boundary for stop of training, default=%.2f" % STOP_REWARD)
    args = parser.parse_args()
    device = torch.device("cuda")

    env = gym.make(DEFAULT_ENV_NAME)
    env = wrappers.wrap_dqn(env)

    net = dqn_model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)

    date_time = datetime.datetime.now().strftime('%d-%b-%Y_%X_%f')
    run_name = f'{DEFAULT_ENV_NAME}_{date_time}'
    writer = SummaryWriter('runs/' + run_name)

    agent = Agent(net, device=device)
    exp_source = experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS_DEFAULT)
    buffer = experience.ExperienceReplayBuffer(exp_source, buffer_size=REPLAY_SIZE)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    frame_idx = 0
    with RewardTracker(writer, net, run_name) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) < REPLAY_START_SIZE:
                continue

            optimizer.zero_grad()
            batch = buffer.sample(BATCH_SIZE)
            loss_v = calc_loss_dqn(batch, net, tgt_net, gamma=GAMMA**REWARD_STEPS_DEFAULT,
                                    device=device, double=DOUBLE_DQN)
            loss_v.backward()
            optimizer.step()

            if frame_idx % SYNC_TARGET_FRAMES == 0:
                tgt_net.load_state_dict(net.state_dict())
    
