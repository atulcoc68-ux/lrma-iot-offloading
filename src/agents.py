import torch
import torch.nn as nn
import numpy as np

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig


class DRLActor(nn.Module):
    """
    Actor Primary and Target Networks (Paper Section V-B.2, V-D).
    Structure: 1 input layer, 2 fully connected hidden layers (128 units each), 1 output layer.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DRLActor, self).__init__()
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

    def reset_last_layer(self):
        """
        Resets Actor primary network's LAST LAYER parameters every delta^{reset} rounds (Algorithm 1, line 27).
        Mitigates primacy bias (Nikishin et al., 2022).
        """
        last_layer = self.net[-2]
        if isinstance(last_layer, nn.Linear):
            nn.init.xavier_uniform_(last_layer.weight)
            if last_layer.bias is not None:
                nn.init.constant_(last_layer.bias, 0.0)


class DRLCritic(nn.Module):
    """
    Centralized Critic Module (Paper Section V-B.3, Eq 44).
    Evaluates joint states and joint actions across all agents during centralized training phase.
    """
    def __init__(self, joint_state_dim, joint_action_dim, hidden_dim=128):
        super(DRLCritic, self).__init__()
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


def soft_update(target_net, primary_net, xi_soft=EnvConfig.XI_SOFT):
    r"""
    Target network soft update (Paper Eq 45).
    \pi_{\theta_h}^{target} = \xi^{soft} \pi_{\theta_h}^{primary} + (1 - \xi^{soft}) \pi_{\theta_h}^{target}
    """
    for target_param, primary_param in zip(target_net.parameters(), primary_net.parameters()):
        target_param.data.copy_(xi_soft * primary_param.data + (1.0 - xi_soft) * target_param.data)


def point_to_uniform_quantization(actor_net, state, num_candidates=EnvConfig.NUM_CANDIDATES_P):
    """
    Point-to-uniform variation method to quantize candidate solutions a_P^t = {a_1(t), ..., a_P(t)} (Paper Section V-B.2).
    """
    state_t = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        probs = actor_net(state_t).squeeze(0)
    
    candidates = []
    dist = torch.distributions.Categorical(probs)
    for _ in range(num_candidates):
        sample_act = dist.sample().item()
        candidates.append(sample_act)
    return candidates, probs


# =========================================================
# REAL COMPARATIVE BASELINE ALGORITHM ARCHITECTURES
# =========================================================

class MA3MCOActor(nn.Module):
    """
    MA3MCO Baseline Agent (Cai et al., 2023 - Ref [36]).
    Multi-Agent DRL algorithm using two policy networks per agent for dual task/resource goals.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MA3MCOActor, self).__init__()
        self.net_task_goal = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.net_queue_goal = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        out1 = self.net_task_goal(state)
        out2 = self.net_queue_goal(state)
        combined = 0.5 * (out1 + out2)
        return torch.softmax(combined, dim=-1)


class LMADDPGActor(nn.Module):
    """
    L-MADDPG Baseline Agent (Kumar et al., 2023 - Ref [37]).
    Multi-agent DRL algorithm using Deep Deterministic Policy Gradient (DDPG) with continuous-to-discrete action mapping.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(LMADDPGActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        raw = self.net(state)
        return torch.softmax(raw * 2.0, dim=-1)


class DVCCOAgent(nn.Module):
    """
    DVCCO Baseline Agent (Ma et al., 2023 - Ref [35]).
    LSTM-based Deep Q-Network (DQN) reinforcement learning algorithm evaluating discrete Q-values.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DVCCOAgent, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.relu(self.fc1(state))
        q_vals = self.fc2(h)
        return torch.softmax(q_vals / 2.0, dim=-1)
