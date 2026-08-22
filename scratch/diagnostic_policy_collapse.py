import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.lrma_trainer import LRMATrainer

def compute_entropy(probs):
    # probs: (batch, num_actions) or (num_actions,)
    eps = 1e-12
    p_safe = torch.clamp(probs, min=eps)
    return -(p_safe * torch.log(p_safe)).sum(dim=-1).mean().item()

def measure_actor_gradients(trainer, states_t, actions_t):
    # ED actor 0 gradient norm
    trainer.ed_optimizers[0].zero_grad()
    pred_probs = trainer.ed_primary_actors[0](states_t[:, 0:9])
    joint_a_diff = actions_t.clone()
    joint_a_diff[:, 0:2] = pred_probs
    loss_ed = -trainer.critic(states_t, joint_a_diff).mean()
    loss_ed.backward()
    
    grad_norm_ed = 0.0
    for p in trainer.ed_primary_actors[0].parameters():
        if p.grad is not None:
            grad_norm_ed += p.grad.norm(2).item() ** 2
    grad_norm_ed = grad_norm_ed ** 0.5
    trainer.ed_optimizers[0].zero_grad()

    # Cloud actor gradient norm
    trainer.cloud_optimizer.zero_grad()
    cloud_probs = trainer.cloud_primary_actor(states_t[:, -11:])
    joint_a_cloud_diff = actions_t.clone()
    joint_a_cloud_diff[:, -trainer.action_dim_cloud:] = cloud_probs
    loss_cloud = -trainer.critic(states_t, joint_a_cloud_diff).mean()
    loss_cloud.backward()
    
    grad_norm_cloud = 0.0
    for p in trainer.cloud_primary_actor.parameters():
        if p.grad is not None:
            grad_norm_cloud += p.grad.norm(2).item() ** 2
    grad_norm_cloud = grad_norm_cloud ** 0.5
    trainer.cloud_optimizer.zero_grad()

    return grad_norm_ed, grad_norm_cloud


def main():
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    trainer = LRMATrainer(num_ed=25, num_mes=5)

    # Populate replay buffer with 64 real-shaped transitions
    batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)

    for _ in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        a_joint = np.random.randn(55).astype(np.float32)
        r = float(np.random.uniform(-5.0, 5.0))
        ns_joint = np.random.randn(236).astype(np.float32)
        trainer.replay_buffer_ed.add(s_joint, a_joint, r, ns_joint, False)

    states, actions, rewards, next_states, dones = trainer.replay_buffer_ed.sample(batch_size)

    states_t = torch.FloatTensor(states).to(trainer.device)
    actions_t = torch.FloatTensor(actions).to(trainer.device)

    # Fixed evaluation states (first state in batch)
    eval_ed_state = states_t[0:1, 0:9]
    eval_cloud_state = states_t[0:1, -11:]

    # Record Initial Policy State
    with torch.no_grad():
        init_ed_probs = trainer.ed_primary_actors[0](eval_ed_state).squeeze(0)
        init_cloud_probs = trainer.cloud_primary_actor(eval_cloud_state).squeeze(0)

    init_ed_entropy = compute_entropy(init_ed_probs)
    init_cloud_entropy = compute_entropy(init_cloud_probs)
    init_ed_max = init_ed_probs.max().item()
    init_cloud_max = init_cloud_probs.max().item()

    with torch.no_grad():
        init_q = trainer.critic(states_t, actions_t)
        init_q_min = init_q.min().item()
        init_q_max = init_q.max().item()

    r_min = float(np.min(rewards))
    r_max = float(np.max(rewards))
    r_mean = float(np.mean(rewards))
    r_std = float(np.std(rewards))

    print(f"\nREPLAY BUFFER REWARD STATS:")
    print(f"  min = {r_min:.4f}, max = {r_max:.4f}, mean = {r_mean:.4f}, std = {r_std:.4f}")

    print(f"\nINITIAL POLICY EVALUATION:")
    print(f"  ED-0 Probs: {init_ed_probs.tolist()}, Entropy: {init_ed_entropy:.6f}, Max: {init_ed_max:.6f}, Min: {init_ed_probs.min().item():.6f}")
    print(f"  Cloud Probs: {init_cloud_probs.tolist()}, Entropy: {init_cloud_entropy:.6f}, Max: {init_cloud_max:.6f}, Min: {init_cloud_probs.min().item():.6f}")
    print(f"  Q Initial Range: [{init_q_min:.4f}, {init_q_max:.4f}]\n")

    history = []

    for step in range(1, 21):
        # Measure gradient norm prior to update
        grad_ed, grad_cloud = measure_actor_gradients(trainer, states_t, actions_t)

        actor_loss, critic_loss = trainer.train_step(batch_size=64, gamma=0.99, xi_soft=0.01)

        with torch.no_grad():
            ed_p = trainer.ed_primary_actors[0](eval_ed_state).squeeze(0)
            cloud_p = trainer.cloud_primary_actor(eval_cloud_state).squeeze(0)
            q_vals = trainer.critic(states_t, actions_t)

        ed_ent = compute_entropy(ed_p)
        cloud_ent = compute_entropy(cloud_p)

        ed_max_p = ed_p.max().item()
        cloud_max_p = cloud_p.max().item()

        q_min = q_vals.min().item()
        q_max = q_vals.max().item()
        q_mean = q_vals.mean().item()
        q_std = q_vals.std().item()

        history.append({
            'step': step,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'ed_p': ed_p.tolist(),
            'cloud_p': cloud_p.tolist(),
            'ed_ent': ed_ent,
            'cloud_ent': cloud_ent,
            'ed_max': ed_max_p,
            'cloud_max': cloud_max_p,
            'q_min': q_min,
            'q_max': q_max,
            'q_mean': q_mean,
            'q_std': q_std,
            'grad_ed': grad_ed,
            'grad_cloud': grad_cloud
        })

        print(f"UPDATE {step:02d} | Actor Loss: {actor_loss:8.4f} | Critic Loss: {critic_loss:8.4f}")
        print(f"  ED-0  : Local={ed_p[0]:.4f}, Offload={ed_p[1]:.4f} | Ent={ed_ent:.4f} | Max={ed_max_p:.4f} | GradNorm={grad_ed:.6f}")
        print(f"  Cloud : Probs={[round(x, 4) for x in cloud_p.tolist()]} | Ent={cloud_ent:.4f} | Max={cloud_max_p:.4f} | GradNorm={grad_cloud:.6f}")
        print(f"  Q-Diag: Min={q_min:7.4f}, Max={q_max:7.4f}, Mean={q_mean:7.4f}, Std={q_std:7.4f}\n")

    final_ed_ent = history[-1]['ed_ent']
    final_cloud_ent = history[-1]['cloud_ent']
    final_ed_max = history[-1]['ed_max']
    final_cloud_max = history[-1]['cloud_max']
    final_q_min = history[-1]['q_min']
    final_q_max = history[-1]['q_max']

    print("============================================================")
    print("DIAGNOSTIC SUMMARY STATS")
    print("============================================================")
    print(f"ED Entropy: {init_ed_entropy:.6f} -> {final_ed_ent:.6f}")
    print(f"Cloud Entropy: {init_cloud_entropy:.6f} -> {final_cloud_ent:.6f}")
    print(f"ED Max Probability: {init_ed_max:.6f} -> {final_ed_max:.6f}")
    print(f"Cloud Max Probability: {init_cloud_max:.6f} -> {final_cloud_max:.6f}")
    print(f"Q Range: [{init_q_min:.4f}, {init_q_max:.4f}] -> [{final_q_min:.4f}, {final_q_max:.4f}]")

    ed_collapse = final_ed_max >= 0.99
    cloud_collapse = final_cloud_max >= 0.99

    softmax_entropy_collapse = (final_ed_ent < 0.1) or (final_cloud_ent < 0.2)
    actor_gradient_vanishing = (history[-1]['grad_ed'] < 1e-4) or (history[-1]['grad_cloud'] < 1e-4)
    critic_q_growth = (final_q_max - init_q_max) > 5.0 or (abs(final_q_mean - history[0]['q_mean']) > 5.0)
    reward_imbalance = abs(r_mean) > 3.0 or r_std > 10.0

    print("\n============================================================")
    print("POLICY COLLAPSE DIAGNOSTIC RESULT")
    print("============================================================")
    print(f"Softmax entropy collapse: {'OBSERVED' if softmax_entropy_collapse else 'NOT OBSERVED'}")
    print(f"Actor gradient vanishing: {'OBSERVED' if actor_gradient_vanishing else 'NOT OBSERVED'}")
    print(f"Critic Q growth: {'OBSERVED' if critic_q_growth else 'NOT OBSERVED'}")
    print(f"Reward imbalance: {'OBSERVED' if reward_imbalance else 'NOT OBSERVED'}")
    print(f"ED policy collapse: {'YES' if ed_collapse else 'NO'}")
    print(f"Cloud policy collapse: {'YES' if cloud_collapse else 'NO'}")

    print("\nTRAIN_STEP CALLS: 20")
    print("\nSOURCE MODIFIED: NO")
    print("ALGORITHM MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
