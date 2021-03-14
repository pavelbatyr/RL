import numpy as np
import torch


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array(exp_list, dtype=np.float32)
    return torch.tensor(np_states, dtype=torch.float)


class Agent:
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and converts them into the actions
    """
    def __init__(self, dqn_model, device="cpu", preprocessor=default_states_preprocessor):
        self.dqn_model = dqn_model
        self.device = device
        self.preprocessor = preprocessor

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    @torch.no_grad()
    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)

        q_v = self.dqn_model(states)
        q_scores = q_v.data.cpu().numpy()

        batch_size, n_actions = q_scores.shape
        # greedy action selection
        actions = np.argmax(q_scores, axis=1)

        return actions, agent_states
