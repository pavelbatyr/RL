import numpy as np
import torch


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
    s = np.array(states, copy=False)
    a = np.array(actions)
    r = np.array(rewards, dtype=np.float32)
    d = np.array(dones, dtype=np.float32)
    l_s = np.array(last_states, copy=False)
    return s, a, r, d, l_s


def evaluate_quantile_at_action(s_quantiles, actions):
    assert s_quantiles.shape[0] == actions.shape[0]

    batch_size = s_quantiles.shape[0]
    N = s_quantiles.shape[1]

    action_index = actions[..., None].expand(batch_size, N, 1)
    sa_quantiles = s_quantiles.gather(dim=2, index=action_index)

    return sa_quantiles


def calculate_huber_loss(td_errors, kappa=1.0):
    kappas = torch.full_like(td_errors, kappa)
    td_errors_abs = td_errors.abs()
    return torch.where(
        td_errors_abs <= kappas,
        0.5 * td_errors ** 2,
        kappas * (td_errors_abs - 0.5 * kappas))


def calculate_quantile_huber_loss(td_errors, taus, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (batch_size, N, N_dash)

    deltas = taus[..., None] - (td_errors.detach() < 0).float()
    element_wise_quantile_huber_loss = \
        torch.abs(deltas) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (batch_size, N, N_dash)

    batch_quantile_huber_loss = \
        element_wise_quantile_huber_loss.sum(dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    quantile_huber_loss = batch_quantile_huber_loss.mean()
    return quantile_huber_loss, batch_quantile_huber_loss.squeeze(1) + 1e-5
