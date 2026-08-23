import torch
import torch.nn as nn
import numpy as np


class EDActor(nn.Module):
    """
    ED Primary and Target Actor Network (IEEE TNSE 2025 Paper Section V-B.1, Eq. 34 & 36).
    Input: ED local state S_{i,k}(t) (dim = 9).
    Output: Softmax probabilities over binary offloading decision x_{i,k}^t in {0, 1} (dim = 2).
            0 = Local execution, 1 = Offloading intent.
    Architecture: 1 input layer (9), 2 FC hidden layers (128 units, ReLU), 1 output layer (2, Softmax).
    """
    def __init__(self, state_dim=9, action_dim=2, hidden_dim=128):
        super(EDActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def reset_last_layer_parameters(self):
        """
        Resets ONLY the primary Actor network's last layer parameters every delta^{reset} = 50 slots (Algorithm 1, line 27).
        """
        last_layer = self.net[4]
        if isinstance(last_layer, nn.Linear):
            nn.init.xavier_uniform_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.0)

    def reset_parameters(self):
        """Resets primary actor network's last layer parameters (Algorithm 1, line 27)."""
        self.reset_last_layer_parameters()


class CloudActor(nn.Module):
    """
    Cloud Primary and Target Actor Network (IEEE TNSE 2025 Paper Section V-B.1, Eq. 35 & 37).
    Input: Cloud state S_{i,k}^{cloud}(t) (dim = 11).
    Output: Softmax probabilities over MES node allocation y_{i,k,j}^t in {0, 1, 2, 3, 4} (dim = M = 5).
    Architecture: 1 input layer (11), 2 FC hidden layers (128 units, ReLU), 1 output layer (5, Softmax).
    """
    def __init__(self, state_dim=11, action_dim=5, hidden_dim=128):
        super(CloudActor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

    def reset_last_layer_parameters(self):
        """
        Resets ONLY the primary Cloud Actor network's last layer parameters every delta^{reset} = 50 slots (Algorithm 1, line 27).
        """
        last_layer = self.net[4]
        if isinstance(last_layer, nn.Linear):
            nn.init.xavier_uniform_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.0)

    def reset_parameters(self):
        """Resets primary Cloud actor network's last layer parameters (Algorithm 1, line 27)."""
        self.reset_last_layer_parameters()


class CentralizedCritic(nn.Module):
    """
    Centralized Critic Module (IEEE TNSE 2025 Paper Section V-B.3, Eq. 44).
    Evaluates joint states S(t) and joint actions a(t) across all agents during CTDE.
    For N=25 EDs and M=5 MES nodes:
        - Joint State Dim = 25 * 9 + 11 = 236.
        - Joint Action Dim = 25 * 2 (ED action probabilities/one-hot) + 5 (Cloud action probabilities/one-hot) = 55.
        - Total Input Dim = 236 + 55 = 291.
    Architecture: FC(joint_input_dim, 128) -> ReLU -> FC(128, 128) -> ReLU -> FC(128, 1).
    """
    def __init__(self, joint_state_dim, joint_action_dim, hidden_dim=128):
        super(CentralizedCritic, self).__init__()
        self.joint_state_dim = joint_state_dim
        self.joint_action_dim = joint_action_dim
        self.net = nn.Sequential(
            nn.Linear(joint_state_dim + joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, joint_state, joint_action):
        x = torch.cat([joint_state, joint_action], dim=-1)
        return self.net(x)

    def reset_parameters(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


def soft_update(target_net, primary_net, xi_soft=0.01):
    r"""
    Target network soft update (Paper Eq. 45):
    theta^{target} = xi^{soft} * theta^{primary} + (1 - xi^{soft}) * theta^{target}
    """
    for target_param, primary_param in zip(target_net.parameters(), primary_net.parameters()):
        target_param.data.copy_(xi_soft * primary_param.data + (1.0 - xi_soft) * target_param.data)
