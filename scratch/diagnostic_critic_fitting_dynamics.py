import sys
import os
import copy
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


def param_norm(model):
    norm_sq = 0.0
    for p in model.parameters():
        norm_sq += p.norm(2).item() ** 2
    return norm_sq ** 0.5


def model_distance(model1, model2):
    dist_sq = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        dist_sq += (p1 - p2).norm(2).item() ** 2
    return dist_sq ** 0.5


def run_critic_experiment(lr_factor, base_lr, device_str):
    target_lr = base_lr * lr_factor
    mode_label = f"LR={target_lr:.6f} (Factor {lr_factor})"
    print(f"\n============================================================")
    print(f"RUNNING CRITIC FITTING EXPERIMENT: {mode_label}")
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

    # Deep copies of models and custom optimizer for isolation
    critic_local = copy.deepcopy(trainer.critic).to(trainer.device)
    critic_trg_local = copy.deepcopy(trainer.critic_target).to(trainer.device)
    opt_local = torch.optim.Adam(critic_local.parameters(), lr=target_lr)

    history = []
    max_c_dist = model_distance(critic_local, critic_trg_local)
    max_grad_norm = 0.0

    for step in range(1, 21):
        # 1. Target Q & TD Target calculation
        with torch.no_grad():
            t_ed_actions = []
            for idx in range(trainer.num_ed):
                ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                t_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
            cloud_st = next_states_t[:, -11:]
            t_cloud_probs = trainer.cloud_target_actor(cloud_st)
            t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

            tQ = critic_trg_local(next_states_t, t_joint_action)
            ty = rewards_t + (1.0 - dones_t) * 0.99 * tQ

        # 2. Current Q & Loss
        cQ = critic_local(states_t, actions_t)
        loss = nn.MSELoss()(cQ, ty)

        opt_local.zero_grad()
        loss.backward()

        c_grad_norm = 0.0
        for p in critic_local.parameters():
            if p.grad is not None:
                c_grad_norm += p.grad.norm(2).item() ** 2
        c_grad_norm = c_grad_norm ** 0.5
        if c_grad_norm > max_grad_norm:
            max_grad_norm = c_grad_norm

        opt_local.step()
        soft_update(critic_trg_local, critic_local, 0.01)

        c_dist = model_distance(critic_local, critic_trg_local)
        if c_dist > max_c_dist:
            max_c_dist = c_dist

        # 3. Action Sensitivity & ED-0 Preferences
        a_grad_in = actions_t.clone().detach().requires_grad_(True)
        q_for_grad = critic_local(states_t, a_grad_in)
        q_for_grad.sum().backward()
        dq_da = a_grad_in.grad.detach()
        dq_da_norm = float(dq_da.norm(2).item())

        dq_da_local = dq_da[:, 0].mean().item()
        dq_da_offload = dq_da[:, 1].mean().item()
        dq_da_diff = dq_da_offload - dq_da_local

        # Direct Action Evaluation for ED-0: Local [1, 0] vs Offload [0, 1]
        a_ed0_local = actions_t.clone()
        a_ed0_local[:, 0] = 1.0
        a_ed0_local[:, 1] = 0.0

        a_ed0_offload = actions_t.clone()
        a_ed0_offload[:, 0] = 0.0
        a_ed0_offload[:, 1] = 1.0

        with torch.no_grad():
            q_eval_loc = critic_local(states_t, a_ed0_local)
            q_eval_off = critic_local(states_t, a_ed0_offload)

        q_eval_diff = (q_eval_off - q_eval_loc).mean().item()

        scQ = get_stats(cQ)
        stQ = get_stats(tQ)
        sty = get_stats(ty)

        history.append({
            'step': step,
            'cQ': scQ,
            'tQ': stQ,
            'ty': sty,
            'loss': loss.item(),
            'grad_norm': c_grad_norm,
            'c_dist': c_dist,
            'dq_da_norm': dq_da_norm,
            'dq_da_diff': dq_da_diff,
            'q_eval_diff': q_eval_diff
        })

        print(f"Step {step:02d} | Loss: {loss.item():7.4f} | Grad Norm: {c_grad_norm:7.4f} | Dist: {c_dist:7.4f}")
        print(f"  Current Q : min={scQ['min']:7.4f}, max={scQ['max']:7.4f}, mean={scQ['mean']:7.4f}, std={scQ['std']:7.4f}")
        print(f"  Target Q  : min={stQ['min']:7.4f}, max={stQ['max']:7.4f}, mean={stQ['mean']:7.4f}, std={stQ['std']:7.4f}")
        print(f"  dQ/dA Diff: {dq_da_diff:+.6f} | Q(offload)-Q(local): {q_eval_diff:+.6f}\n")

    return {
        'lr_factor': lr_factor,
        'target_lr': target_lr,
        'final_cQ_max': history[-1]['cQ']['max'],
        'final_cQ_range': (history[-1]['cQ']['min'], history[-1]['cQ']['max']),
        'final_loss': history[-1]['loss'],
        'final_dist': history[-1]['c_dist'],
        'max_dist': max_c_dist,
        'max_grad_norm': max_grad_norm,
        'final_dq_da_diff': history[-1]['dq_da_diff'],
        'final_q_eval_diff': history[-1]['q_eval_diff'],
        'history': history
    }


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

    base_lr = 0.001
    res_A = run_critic_experiment(1.0, base_lr, device_str)      # LR = 0.001
    res_B = run_critic_experiment(0.1, base_lr, device_str)      # LR = 0.0001
    res_C = run_critic_experiment(0.01, base_lr, device_str)     # LR = 0.00001

    print("\n============================================================")
    print("CRITIC FITTING DYNAMICS EXPERIMENT COMPARISON TABLE")
    print("============================================================")
    print(f"{'LR Factor':<10} | {'LR Value':<10} | {'Final Loss':<12} | {'Final Dist':<12} | {'Max Grad Norm':<14} | {'Q(off)-Q(loc)':<14} | {'dQ/dA Diff':<14}")
    print("-" * 105)

    for r in [res_A, res_B, res_C]:
        print(f"{r['lr_factor']:<10.2f} | {r['target_lr']:<10.6f} | {r['final_loss']:<12.6f} | {r['final_dist']:<12.6f} | {r['max_grad_norm']:<14.6f} | {r['final_q_eval_diff']:<+14.6f} | {r['final_dq_da_diff']:<+14.6f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    q_growth_reduced_by_lr = (res_C['final_cQ_max'] < res_A['final_cQ_max'] * 0.5)
    loss_stable = (res_A['final_loss'] < 10.0)
    extrapolation_detected = (res_A['final_q_eval_diff'] > 1e-3 and res_A['final_dq_da_diff'] > 1e-3)
    target_lag_detected = (res_A['final_dist'] > 2.0)

    if q_growth_reduced_by_lr and res_C['final_loss'] < res_A['final_loss']:
        classification = "A. CRITIC LEARNING RATE IS TOO AGGRESSIVE"
    elif extrapolation_detected:
        classification = "B. CRITIC EXTRAPOLATES BETWEEN ONE-HOT ACTIONS"
    elif target_lag_detected:
        classification = "D. CRITIC-TARGET LAG IS THE PRIMARY CAUSE"
    elif loss_stable:
        classification = "C. CRITIC FITTING IS STABLE BUT ACTION PREFERENCE IS WRONG"
    else:
        classification = "E. NO CLEAR CRITIC-FITTING FAILURE IDENTIFIED"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
