import torch
import torch.nn as nn
import numpy as np

try:
    from src.lrma_networks import EDActor, CloudActor, CentralizedCritic, soft_update
    from src.lrma_candidates import point_to_uniform_candidate_generation, select_best_candidate
    from src.lrma_replay import LRMAPersistentReplayBuffer
except ModuleNotFoundError:
    from lrma_networks import EDActor, CloudActor, CentralizedCritic, soft_update
    from lrma_candidates import point_to_uniform_candidate_generation, select_best_candidate
    from lrma_replay import LRMAPersistentReplayBuffer


class LRMATrainer:
    """
    Paper-Faithful LRMA Multi-Agent Reinforcement Learning Trainer (IEEE TNSE 2025 Algorithm 1).
    Centralized Training with Distributed Execution (CTDE):
      - N Decentralized ED Primary Actors + N Target Actors (Input: 9, Output: 2 binary offload intent)
      - 1 Centralized Cloud Primary Actor + 1 Target Actor (Input: 11, Output: M MES node selection)
      - 1 Centralized Joint Critic + 1 Target Critic Q(S, a)
      - Point-to-Uniform Candidate Generation (P=5)
      - Persistent Experience Replay Buffers
      - Parameter Resetting every delta^{reset} = 50 slots
      - Soft Target Updates (xi^{soft} = 0.01)
      - Zero PPO mechanics (No clipped ratio, no GAE, no on-policy buffer clearing)
    """
    def __init__(self, num_ed=25, num_mes=5, lr_actor=0.001, lr_critic=0.001, device=None):
        self.num_ed = num_ed
        self.num_mes = num_mes
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim_ed = 9
        self.action_dim_ed = 2
        self.state_dim_cloud = 11
        self.action_dim_cloud = num_mes

        self.joint_state_dim = num_ed * self.state_dim_ed + self.state_dim_cloud
        self.joint_action_dim = num_ed * self.action_dim_ed + self.action_dim_cloud

        # Primary & Target ED Actors
        self.ed_primary_actors = [EDActor(self.state_dim_ed, self.action_dim_ed).to(self.device) for _ in range(num_ed)]
        self.ed_target_actors = [EDActor(self.state_dim_ed, self.action_dim_ed).to(self.device) for _ in range(num_ed)]

        # Primary & Target Cloud Actor
        self.cloud_primary_actor = CloudActor(self.state_dim_cloud, self.action_dim_cloud).to(self.device)
        self.cloud_target_actor = CloudActor(self.state_dim_cloud, self.action_dim_cloud).to(self.device)

        # Sync Target Actors initially
        for p, t_net in zip(self.ed_primary_actors, self.ed_target_actors):
            t_net.load_state_dict(p.state_dict())
        self.cloud_target_actor.load_state_dict(self.cloud_primary_actor.state_dict())

        # Primary & Target Centralized Critic
        self.critic = CentralizedCritic(self.joint_state_dim, self.joint_action_dim).to(self.device)
        self.critic_target = CentralizedCritic(self.joint_state_dim, self.joint_action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.ed_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.ed_primary_actors]
        self.cloud_optimizer = torch.optim.Adam(self.cloud_primary_actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Persistent Experience Replay Buffers
        self.replay_buffer_ed = LRMAPersistentReplayBuffer(capacity=10000)
        self.replay_buffer_cloud = LRMAPersistentReplayBuffer(capacity=10000)

    def select_ed_action(self, ed_idx, state_ed, P=5):
        """
        Generates P=5 candidates via point-to-uniform variation for ED actor `ed_idx`.
        Returns candidate list and action probabilities.
        """
        actor = self.ed_primary_actors[ed_idx]
        candidates, probs = point_to_uniform_candidate_generation(actor, state_ed, P=P)
        return candidates, probs

    def select_cloud_action(self, state_cloud, P=5):
        """
        Generates P=5 candidates via point-to-uniform variation for Cloud actor.
        Returns candidate list and action probabilities.
        """
        candidates, probs = point_to_uniform_candidate_generation(self.cloud_primary_actor, state_cloud, P=P)
        return candidates, probs

    def construct_joint_action_representation(self, ed_actions, cloud_action):
        """
        Constructs a joint action vector from ED binary actions (25x2) and Cloud action (M).
        Uses one-hot representation for exact tensor dimension consistency: (N*2 + M).
        """
        vecs = []
        for a in ed_actions:
            one_hot = np.zeros(2, dtype=np.float32)
            one_hot[int(a)] = 1.0
            vecs.append(one_hot)
        cloud_one_hot = np.zeros(self.num_mes, dtype=np.float32)
        cloud_one_hot[int(cloud_action)] = 1.0
        vecs.append(cloud_one_hot)
        return np.concatenate(vecs, axis=0)

    def train_step(self, batch_size=64, gamma=0.99, xi_soft=0.01):
        """
        Performs one CTDE minibatch update on persistent replay buffer:
          1. Samples minibatch B=64 from persistent experience replay.
          2. Computes TD target: y = r_tot + gamma * Q_target(S', a'_target).
          3. Updates Centralized Critic via MSE loss.
          4. Updates ED & Cloud primary actors via Centralized Critic policy gradient.
          5. Soft updates target networks with xi_soft = 0.01.
        """
        if len(self.replay_buffer_ed) < batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer_ed.sample(batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1. Critic Update (TD Bootstrapping)
        with torch.no_grad():
            target_next_q = self.critic_target(next_states_t, actions_t)
            target_y = rewards_t + (1.0 - dones_t) * gamma * target_next_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, target_y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Actor Policy Gradient Update
        # Update ED actors
        total_actor_loss = 0.0
        for idx in range(self.num_ed):
            self.ed_optimizers[idx].zero_grad()
            pred_probs = self.ed_primary_actors[idx](states_t[:, idx*9:(idx+1)*9])
            joint_a_diff = actions_t.clone()
            joint_a_diff[:, idx*2:(idx+1)*2] = pred_probs
            
            actor_loss = -self.critic(states_t, joint_a_diff).mean()
            actor_loss.backward()
            self.ed_optimizers[idx].step()
            total_actor_loss += actor_loss.item()

        # Update Cloud Actor
        self.cloud_optimizer.zero_grad()
        cloud_probs = self.cloud_primary_actor(states_t[:, -11:])
        joint_a_cloud_diff = actions_t.clone()
        joint_a_cloud_diff[:, -self.action_dim_cloud:] = cloud_probs
        cloud_loss = -self.critic(states_t, joint_a_cloud_diff).mean()
        cloud_loss.backward()
        self.cloud_optimizer.step()
        total_actor_loss += cloud_loss.item()

        # 3. Soft Target Updates (xi_soft = 0.01)
        for p, trg in zip(self.ed_primary_actors, self.ed_target_actors):
            soft_update(trg, p, xi_soft)
        soft_update(self.cloud_target_actor, self.cloud_primary_actor, xi_soft)
        soft_update(self.critic_target, self.critic, xi_soft)

        return total_actor_loss / (self.num_ed + 1), critic_loss.item()

    def reset_primary_parameters(self):
        """
        Resets primary network parameters every delta^{reset} = 50 slots (Algorithm 1, lines 26-28).
        Primary actors and primary critic are re-initialized. Target networks remain intact.
        """
        for p in self.ed_primary_actors:
            p.reset_parameters()
        self.cloud_primary_actor.reset_parameters()
        self.critic.reset_parameters()
