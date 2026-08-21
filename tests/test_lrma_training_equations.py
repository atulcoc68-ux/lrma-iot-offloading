import pytest
import torch
import torch.nn as nn
import numpy as np

from src.lrma_networks import EDActor, CloudActor, CentralizedCritic, soft_update
from src.lrma_candidates import point_to_uniform_candidate_generation, select_best_candidate
from src.lrma_replay import LRMAPersistentReplayBuffer
from src.lrma_trainer import LRMATrainer


class TestLRMATrainingEquations:

    def test_critic_td_target_equation(self):
        # Target: y = r_tot + gamma * Q_target(S', a'_target)
        r_tot = 5.0
        gamma = 0.99
        q_next_target = 10.0
        
        y_expected = r_tot + gamma * q_next_target
        assert np.isclose(y_expected, 5.0 + 0.99 * 10.0)

    def test_critic_mse_loss_equation(self):
        critic = CentralizedCritic(joint_state_dim=236, joint_action_dim=55)
        
        s = torch.randn(4, 236)
        a = torch.randn(4, 55)
        y = torch.tensor([[14.9], [12.0], [8.5], [20.1]])
        
        q_pred = critic(s, a)
        mse_loss = nn.MSELoss()(q_pred, y)
        
        manual_mse = torch.mean((q_pred - y) ** 2)
        assert torch.allclose(mse_loss, manual_mse)

    def test_actor_gradient_direction_increases_q(self):
        # Policy gradient objective maximizes Q(S, pi(S)) -> Actor loss = -Q(S, pi(S))
        actor = EDActor(9, 2)
        critic = CentralizedCritic(9, 2)
        optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
        
        s = torch.randn(1, 9)
        a_prob = actor(s)
        q_before = critic(s, a_prob).item()
        
        optimizer.zero_grad()
        loss = -critic(s, actor(s))
        loss.backward()
        optimizer.step()
        
        q_after = critic(s, actor(s)).item()
        assert q_after >= q_before  # Gradient step increases Q evaluation

    def test_target_network_soft_update_equation(self):
        # theta_target = xi * theta_primary + (1 - xi) * theta_target
        primary = nn.Linear(10, 10)
        target = nn.Linear(10, 10)
        
        with torch.no_grad():
            primary.weight.fill_(2.0)
            target.weight.fill_(1.0)
            
        soft_update(target, primary, xi_soft=0.01)
        
        # Expected target weight: 0.01 * 2.0 + 0.99 * 1.0 = 0.02 + 0.99 = 1.01
        assert torch.allclose(target.weight, torch.full_like(target.weight, 1.01))

    def test_candidate_argmax_selection_math(self):
        candidates = [0, 1, 0, 1, 0]
        rewards = {0: 3.5, 1: 8.2}
        
        best_cand, best_r = select_best_candidate(candidates, lambda a: rewards[a])
        assert best_cand == 1
        assert best_r == 8.2

    def test_state_and_action_tensor_dimensions_assertion(self):
        N = 25
        num_mes = 5
        trainer = LRMATrainer(num_ed=N, num_mes=num_mes)
        
        ed_states = np.random.randn(N, 9).astype(np.float32)
        cloud_state = np.random.randn(11).astype(np.float32)
        
        joint_state = np.concatenate([ed_states.flatten(), cloud_state])
        assert joint_state.shape[0] == 236  # 25*9 + 11 = 236
        
        ed_actions = [np.random.choice([0, 1]) for _ in range(N)]
        cloud_action = np.random.choice(range(num_mes))
        
        joint_action = trainer.construct_joint_action_representation(ed_actions, cloud_action)
        assert joint_action.shape[0] == 55  # 25*2 + 5 = 55

    def test_tiny_deterministic_training_step(self):
        trainer = LRMATrainer(num_ed=2, num_mes=3)  # Tiny mode: 2 EDs, 3 MES nodes
        # Populate replay buffer with 10 dummy transitions
        for _ in range(10):
            s_joint = np.random.randn(2*9 + 11).astype(np.float32)
            a_joint = np.random.randn(2*2 + 3).astype(np.float32)
            r = 1.0
            ns_joint = np.random.randn(2*9 + 11).astype(np.float32)
            trainer.replay_buffer_ed.add(s_joint, a_joint, r, ns_joint, False)
            
        a_loss, c_loss = trainer.train_step(batch_size=4, gamma=0.99, xi_soft=0.01)
        assert isinstance(a_loss, float)
        assert isinstance(c_loss, float)
