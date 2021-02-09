from collections import namedtuple, deque

import numpy as np
import gym

# from agent import Agent


# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class ExperienceSource:
    """
    Simple n-step experience source using an environment
    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1):
        """
        Create simple experience source
        :param env: environment to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        """
        assert isinstance(env, gym.Env)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        self.env = env
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):     
        state = self.env.reset()
        agent_state = self.agent.initial_state()
        history = deque(maxlen=self.steps_count)
        cur_reward = 0.0
        cur_step = 0
        
        iter_idx = 0
        while True:
            states_actions, new_agent_states = self.agent([state], [agent_state])
            action = states_actions[0]
            agent_state = new_agent_states[0]

            next_state, r, is_done, _ = self.env.step(action)

            cur_reward += r
            cur_step += 1
            history.append(Experience(state=state, action=action, reward=r, done=is_done))

            if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                yield tuple(history)
            state = next_state
            if is_done:
                # in case of very short episode (shorter than our steps count), send gathered history
                if history and len(history) < self.steps_count:
                    yield tuple(history)
                # generate tail of history
                while len(history) > 1:
                    history.popleft()
                    yield tuple(history)
                self.total_rewards.append(cur_reward)
                self.total_steps.append(cur_step)

                cur_reward = 0.0
                cur_step = 0
                state = self.env.reset()
                agent_state = self.agent.initial_state()
                history.clear()
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.
    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1):
        assert isinstance(gamma, float)
        super().__init__(env, agent, steps_count+1, steps_delta)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)
