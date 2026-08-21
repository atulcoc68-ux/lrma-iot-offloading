import os
import hashlib
import pytest
import torch
import torch.nn as nn
import numpy as np

from src.lrma_networks import EDActor, CloudActor, CentralizedCritic, soft_update
from src.lrma_candidates import point_to_uniform_candidate_generation, select_best_candidate
from src.lrma_replay import LRMAPersistentReplayBuffer
from src.lrma_trainer import LRMATrainer


class TestLRMATrainingArchitecture:

    def test_ed_actor_dimensions_and_non_shared_parameters(self):
        N = 25
        trainer = LRMATrainer(num_ed=N, num_mes=5)
        
        # 1. Exactly N ED actors
        assert len(trainer.ed_primary_actors) == N
        assert len(trainer.ed_target_actors) == N
        
        # 2. ED actor input = 9, output = 2
        for actor in trainer.ed_primary_actors:
            assert actor.state_dim == 9
            assert actor.action_dim == 2
            device = next(actor.parameters()).device
            x = torch.randn(1, 9, device=device)
            out = actor(x)
            assert out.shape == (1, 2)
            # Softmax outputs sum to 1
            assert torch.allclose(out.sum(dim=-1), torch.tensor([1.0], device=device))

        # 3. ED actors do NOT share parameters
        actor_0_params = list(trainer.ed_primary_actors[0].parameters())
        actor_1_params = list(trainer.ed_primary_actors[1].parameters())
        for p0, p1 in zip(actor_0_params, actor_1_params):
            assert p0 is not p1
            assert not torch.all(p0 == p1)  # Distinct random initializations

    def test_cloud_actor_dimensions(self):
        trainer = LRMATrainer(num_ed=25, num_mes=5)
        cloud_actor = trainer.cloud_primary_actor
        
        # Cloud actor input = 11, output = 5
        assert cloud_actor.state_dim == 11
        assert cloud_actor.action_dim == 5
        device = next(cloud_actor.parameters()).device
        x = torch.randn(1, 11, device=device)
        out = cloud_actor(x)
        assert out.shape == (1, 5)
        assert torch.allclose(out.sum(dim=-1), torch.tensor([1.0], device=device))

    def test_centralized_critic_existence_and_initialization(self):
        N = 25
        trainer = LRMATrainer(num_ed=N, num_mes=5)
        
        # Centralized Critic input: joint_state (236) + joint_action (55) = 291
        assert trainer.critic.joint_state_dim == 236
        assert trainer.critic.joint_action_dim == 55
        
        device = next(trainer.critic.parameters()).device
        s_joint = torch.randn(2, 236, device=device)
        a_joint = torch.randn(2, 55, device=device)
        q_val = trainer.critic(s_joint, a_joint)
        assert q_val.shape == (2, 1)

    def test_target_networks_initialization_and_soft_update(self):
        trainer = LRMATrainer(num_ed=25, num_mes=5)
        
        # Target networks initially match primary networks exactly
        for p, trg in zip(trainer.ed_primary_actors, trainer.ed_target_actors):
            for p_param, trg_param in zip(p.parameters(), trg.parameters()):
                assert torch.allclose(p_param, trg_param)

        cloud_p = trainer.cloud_primary_actor
        cloud_trg = trainer.cloud_target_actor
        for p_param, trg_param in zip(cloud_p.parameters(), cloud_trg.parameters()):
            assert torch.allclose(p_param, trg_param)

        # Mutate primary network and test soft_update
        with torch.no_grad():
            for p_param in cloud_p.parameters():
                p_param.add_(1.0)
                
        soft_update(cloud_trg, cloud_p, xi_soft=0.01)
        for p_param, trg_param in zip(cloud_p.parameters(), cloud_trg.parameters()):
            assert not torch.allclose(p_param, trg_param)  # Target moved by xi=0.01

    def test_persistent_replay_buffer_semantics(self):
        buffer = LRMAPersistentReplayBuffer(capacity=100)
        
        # Add transition
        s = np.zeros(9, dtype=np.float32)
        a = 1
        r = 5.0
        ns = np.ones(9, dtype=np.float32)
        buffer.add(s, a, r, ns, False)
        
        assert len(buffer) == 1
        
        # Sample does NOT clear buffer
        sampled = buffer.sample(batch_size=1)
        assert len(buffer) == 1  # Still persistent!
        assert len(sampled[0]) == 1

    def test_candidate_generation_and_selection(self):
        actor = EDActor(9, 2)
        state = np.random.randn(9).astype(np.float32)
        
        candidates, probs = point_to_uniform_candidate_generation(actor, state, P=5)
        
        # Candidate count = 5
        assert len(candidates) == 5
        for cand in candidates:
            assert cand in [0, 1]
            
        # Select best candidate via argmax reward
        def mock_eval(cand):
            return 10.0 if cand == 1 else 2.0
            
        best_cand, best_r = select_best_candidate(candidates, mock_eval)
        if 1 in candidates:
            assert best_cand == 1
            assert best_r == 10.0

    def test_parameter_reset_mechanism(self):
        trainer = LRMATrainer(num_ed=25, num_mes=5)
        actor_0 = trainer.ed_primary_actors[0]
        
        # Record weights before reset
        old_weights = actor_0.net[0].weight.clone()
        
        # Trigger reset
        trainer.reset_primary_parameters()
        
        new_weights = actor_0.net[0].weight.clone()
        assert not torch.allclose(old_weights, new_weights)

    def test_ppo_components_absent_from_lrma_trainer(self):
        trainer = LRMATrainer(num_ed=25, num_mes=5)
        
        # Verify no PPO attributes exist in trainer
        assert not hasattr(trainer, 'eps_clip')
        assert not hasattr(trainer, 'gae_lambda')
        assert not hasattr(trainer, 'old_log_probs')

    def test_frozen_files_safety(self):
        frozen_files = [
            "src/environment.py",
            "src/lyapunov.py",
            "src/config.py",
            "src/data_loader.py",
            "generate_paper_plots.py"
        ]
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel_path in frozen_files:
            abs_path = os.path.join(base_dir, rel_path)
            assert os.path.exists(abs_path), f"Frozen file missing: {rel_path}"
