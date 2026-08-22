import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.lrma_trainer import LRMATrainer
from src.lrma_networks import soft_update


def get_stats(tensor):
    np_arr = tensor.detach().cpu().numpy()
    return {
        'min': float(np_arr.min()),
        'max': float(np_arr.max()),
        'mean': float(np_arr.mean()),
        'std': float(np_arr.std())
    }


def compute_entropy(probs):
    eps = 1e-12
    p_safe = torch.clamp(probs, min=eps)
    return -(p_safe * torch.log(p_safe)).sum(dim=-1).mean().item()


def compute_dq_da(critic, states_t, actions_t, scale_factor=1.0):
    a_grad = actions_t.clone().detach().requires_grad_(True)
    q_val = critic(states_t, a_grad) / scale_factor
    q_val.sum().backward()
    grad = a_grad.grad.detach()
    
    grad_norm = float(grad.norm(2).item())
    grad_np = grad.cpu().numpy()
    return {
        'norm': grad_norm,
        'min': float(grad_np.min()),
        'max': float(grad_np.max()),
        'mean': float(grad_np.mean()),
        'max_abs': float(np.max(np.abs(grad_np)))
    }


def run_scaling_experiment(use_scaled_critic, device_str):
    mode_str = "SCALED/DIAGNOSTIC" if use_scaled_critic else "NORMAL"
    print(f"\n============================================================")
    print(f"RUNNING EXPERIMENT MODE: {mode_str}")
    print(f"============================================================")

    np.random.seed(42)
    torch.manual_seed(42)

    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 64
    for _ in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        ed_one_hots = []
        for _ in range(25):
            oh = np.zeros(2, dtype=np.float32)
            oh[np.random.choice([0, 1])] = 1.0
            ed_one_hots.append(oh)
        cloud_oh = np.zeros(5, dtype=np.float32)
        cloud_oh[np.random.choice(range(5))] = 1.0

        a_joint = np.concatenate(ed_one_hots + [cloud_oh])
        r = float(np.random.uniform(-5.0, 5.0))
        ns_joint = np.random.randn(236).astype(np.float32)
        trainer.replay_buffer_ed.add(s_joint, a_joint, r, ns_joint, False)

    states, actions, rewards, next_states, dones = trainer.replay_buffer_ed.sample(batch_size)

    states_t = torch.FloatTensor(states).to(trainer.device)
    actions_t = torch.FloatTensor(actions).to(trainer.device)
    next_states_t = torch.FloatTensor(next_states).to(trainer.device)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(trainer.device)
    dones_t = torch.FloatTensor(dones).unsqueeze(1).to(trainer.device)

    eval_ed_state = states_t[0:1, 0:9]
    eval_cloud_state = states_t[0:1, -11:]

    max_dq_da_abs = 0.0
    max_q_val = -1e9
    history = []

    for step in range(1, 21):
        # 1. Target Q & TD Target
        with torch.no_grad():
            t_ed_actions = []
            for idx in range(trainer.num_ed):
                ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                t_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
            cloud_st = next_states_t[:, -11:]
            t_cloud_probs = trainer.cloud_target_actor(cloud_st)
            t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

            tQ = trainer.critic_target(next_states_t, t_joint_action)
            ty = rewards_t + (1.0 - dones_t) * 0.99 * tQ

        # 2. Critic Update
        cQ = trainer.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(cQ, ty)

        trainer.critic_optimizer.zero_grad()
        critic_loss.backward()

        c_grad_norm = 0.0
        for p in trainer.critic.parameters():
            if p.grad is not None:
                c_grad_norm += p.grad.norm(2).item() ** 2
        c_grad_norm = c_grad_norm ** 0.5
        trainer.critic_optimizer.step()

        # Compute dQ/dA diagnostic
        q_scale = max(1.0, float(cQ.detach().std().item())) if use_scaled_critic else 1.0
        dq_da = compute_dq_da(trainer.critic, states_t, actions_t, scale_factor=q_scale)

        if dq_da['max_abs'] > max_dq_da_abs:
            max_dq_da_abs = dq_da['max_abs']

        # 3. Actor Updates
        # ED Actors
        ed0_grad_norm = 0.0
        for idx in range(trainer.num_ed):
            trainer.ed_optimizers[idx].zero_grad()
            pred_probs = trainer.ed_primary_actors[idx](states_t[:, idx * 9 : (idx + 1) * 9])
            joint_a_diff = actions_t.clone()
            joint_a_diff[:, idx * 2 : (idx + 1) * 2] = pred_probs

            critic_out = trainer.critic(states_t, joint_a_diff)
            if use_scaled_critic:
                critic_out = critic_out / q_scale

            actor_loss = -critic_out.mean()
            actor_loss.backward()

            if idx == 0:
                for p in trainer.ed_primary_actors[0].parameters():
                    if p.grad is not None:
                        ed0_grad_norm += p.grad.norm(2).item() ** 2
                ed0_grad_norm = ed0_grad_norm ** 0.5

            trainer.ed_optimizers[idx].step()

        # Cloud Actor
        trainer.cloud_optimizer.zero_grad()
        cloud_probs = trainer.cloud_primary_actor(states_t[:, -11:])
        joint_a_cloud_diff = actions_t.clone()
        joint_a_cloud_diff[:, -trainer.action_dim_cloud:] = cloud_probs

        critic_cloud_out = trainer.critic(states_t, joint_a_cloud_diff)
        if use_scaled_critic:
            critic_cloud_out = critic_cloud_out / q_scale

        cloud_loss = -critic_cloud_out.mean()
        cloud_loss.backward()

        cloud_grad_norm = 0.0
        for p in trainer.cloud_primary_actor.parameters():
            if p.grad is not None:
                cloud_grad_norm += p.grad.norm(2).item() ** 2
        cloud_grad_norm = cloud_grad_norm ** 0.5

        trainer.cloud_optimizer.step()

        # Soft updates
        for p, trg in zip(trainer.ed_primary_actors, trainer.ed_target_actors):
            soft_update(trg, p, 0.01)
        soft_update(trainer.cloud_target_actor, trainer.cloud_primary_actor, 0.01)
        soft_update(trainer.critic_target, trainer.critic, 0.01)

        # Record Context Statistics
        with torch.no_grad():
            ed0_p = trainer.ed_primary_actors[0](eval_ed_state).squeeze(0)
            cloud_p = trainer.cloud_primary_actor(eval_cloud_state).squeeze(0)
            cQ_stats = get_stats(cQ)

        if cQ_stats['max'] > max_q_val:
            max_q_val = cQ_stats['max']

        ed0_ent = compute_entropy(ed0_p)
        cloud_ent = compute_entropy(cloud_p)
        ed0_max_p = ed0_p.max().item()
        cloud_max_p = cloud_p.max().item()

        history.append({
            'step': step,
            'cQ': cQ_stats,
            'c_grad_norm': c_grad_norm,
            'dq_da': dq_da,
            'ed0_ent': ed0_ent,
            'ed0_max': ed0_max_p,
            'ed0_grad': ed0_grad_norm,
            'cloud_ent': cloud_ent,
            'cloud_max': cloud_max_p,
            'cloud_grad': cloud_grad_norm
        })

        print(f"Update {step:02d} [{mode_str}] | Q min/max: {cQ_stats['min']:.4f}/{cQ_stats['max']:.4f} | dQ/dA Norm: {dq_da['norm']:.4f}")
        print(f"  ED-0 : Ent={ed0_ent:.4f}, MaxP={ed0_max_p:.4f}, GradNorm={ed0_grad_norm:.6f}")
        print(f"  Cloud: Ent={cloud_ent:.4f}, MaxP={cloud_max_p:.4f}, GradNorm={cloud_grad_norm:.6f}\n")

    return {
        'mode': mode_str,
        'final_cQ': history[-1]['cQ'],
        'max_q_val': max_q_val,
        'max_dq_da_abs': max_dq_da_abs,
        'final_ed0_ent': history[-1]['ed0_ent'],
        'final_ed0_max': history[-1]['ed0_max'],
        'final_cloud_ent': history[-1]['cloud_ent'],
        'final_cloud_max': history[-1]['cloud_max'],
        'history': history
    }


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

    res_normal = run_scaling_experiment(use_scaled_critic=False, device_str=device_str)
    res_scaled = run_scaling_experiment(use_scaled_critic=True, device_str=device_str)

    print("\n============================================================")
    print("CRITIC SCALE SENSITIVITY REPORT")
    print("============================================================")

    print("\nNORMAL:")
    print(f"  final Q range         : [{res_normal['final_cQ']['min']:.4f}, {res_normal['final_cQ']['max']:.4f}]")
    print(f"  max Q                 : {res_normal['max_q_val']:.4f}")
    print(f"  max |dQ/dA|           : {res_normal['max_dq_da_abs']:.6f}")
    print(f"  ED entropy            : {res_normal['final_ed0_ent']:.6f}")
    print(f"  ED max probability    : {res_normal['final_ed0_max']:.6f}")
    print(f"  Cloud entropy         : {res_normal['final_cloud_ent']:.6f}")
    print(f"  Cloud max probability : {res_normal['final_cloud_max']:.6f}")

    print("\nSCALED/DIAGNOSTIC:")
    print(f"  final Q range         : [{res_scaled['final_cQ']['min']:.4f}, {res_scaled['final_cQ']['max']:.4f}]")
    print(f"  max Q                 : {res_scaled['max_q_val']:.4f}")
    print(f"  max |dQ/dA|           : {res_scaled['max_dq_da_abs']:.6f}")
    print(f"  ED entropy            : {res_scaled['final_ed0_ent']:.6f}")
    print(f"  ED max probability    : {res_scaled['final_ed0_max']:.6f}")
    print(f"  Cloud entropy         : {res_scaled['final_cloud_ent']:.6f}")
    print(f"  Cloud max probability : {res_scaled['final_cloud_max']:.6f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    normal_collapsed = res_normal['final_ed0_max'] >= 0.99
    scaled_collapsed = res_scaled['final_ed0_max'] >= 0.99

    if normal_collapsed and not scaled_collapsed:
        classification = "A. Critic scale strongly contributes to collapse"
    elif res_scaled['final_ed0_ent'] > res_normal['final_ed0_ent'] * 2.0:
        classification = "B. Critic scale likely contributes"
    else:
        classification = "C. Critic scale does not explain collapse"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
