from collections import deque
import math
from typing import Tuple

import numpy as np
import torch as th
from torch import nn


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)

def create_feature_actions(feature_: th.Tensor, action_: th.Tensor):
    N = feature_.size(0)
    # Flatten sequence of features.
    f = feature_[:, :-1].view(N, -1)
    n_f = feature_[:, 1:].view(N, -1)
    # Flatten sequence of actions.
    a = action_[:, :-1].view(N, -1)
    n_a = action_[:, 1:].view(N, -1)
    # Concatenate feature and action.
    fa = th.cat([f, a], dim=-1)
    n_fa = th.cat([n_f, n_a], dim=-1)
    return fa, n_fa


def soft_update(target: nn.Module, source: nn.Module, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def grad_false(network: nn.Module):
    for param in network.parameters():
        param.requires_grad = False


def build_mlp(input_dim: int, output_dim: int, hidden_units=[64, 64], hidden_activation=nn.Tanh(), output_activation=None):
    layers = []
    units = input_dim
    for next_units in hidden_units:
        layers.append(nn.Linear(units, next_units))
        layers.append(hidden_activation)
        units = next_units
    layers.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


def calculate_gaussian_log_prob(log_std: th.Tensor, noise: th.Tensor) -> th.Tensor:
    return (-0.5 * noise.pow(2) - log_std).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_std.size(-1)


def calculate_log_pi(log_std: th.Tensor, noise: th.Tensor, action: th.Tensor) -> th.Tensor:
    gaussian_log_prob = calculate_gaussian_log_prob(log_std, noise)
    return gaussian_log_prob - th.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)


def reparameterize(mean: th.Tensor, log_std: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
    noise = th.randn_like(mean)
    action = th.tanh(mean + noise * log_std.exp())
    return action, calculate_log_pi(log_std, noise, action)


def calculate_kl_divergence(p_mean: th.Tensor, p_std: th.Tensor, q_mean: th.Tensor, q_std: th.Tensor) -> th.Tensor:
    var_ratio = (p_std / q_std).pow_(2)
    t1 = ((p_mean - q_mean) / q_std).pow_(2)
    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
