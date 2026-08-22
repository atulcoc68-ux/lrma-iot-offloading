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


def evaluate_action_landscape(critic, states_t, actions_t):
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    points = []

    q_local_base = None

    for idx, alpha in enumerate(alphas):
        a_test = actions_t.clone().detach().requires_grad_(True)
        # Modify ED-0 action representation to [(1-alpha), alpha]
        with torch.no_grad():
            a_test[:, 0] = 1.0 - alpha
            a_test[:, 1] = alpha

        a_grad_in = a_test.clone().detach().requires_grad_(True)
        q_val = critic(states_t, a_grad_in)
        q_mean_val = q_val.mean()
        q_mean_val.backward()

        stats = get_stats(q_val)
        if alpha == 0.0:
            q_local_base = stats['mean']

        q_diff = stats['mean'] - q_local_base

        dq_da = a_grad_in.grad.detach()
        dq_loc = float(dq_da[:, 0].mean().item())
        dq_off = float(dq_da[:, 1].mean().item())
        dq_diff = dq_off - dq_loc

        points.append({
            'alpha': alpha,
            'stats': stats,
            'q_diff': q_diff,
            'dq_loc': dq_loc,
            'dq_off': dq_off,
            'dq_diff': dq_diff
        })

    # Compute Finite Difference Slopes between adjacent points
    fd_slopes = []
    for i in range(len(points) - 1):
        p_curr = points[i]
        p_next = points[i + 1]
        delta_a = p_next['alpha'] - p_curr['alpha']
        delta_q = p_next['stats']['mean'] - p_curr['stats']['mean']
        fd_slope = delta_q / delta_a
        fd_slopes.append(fd_slope)

    return points, fd_slopes


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

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

    critic = copy.deepcopy(trainer.critic).to(trainer.device)
    critic_trg = copy.deepcopy(trainer.critic_target).to(trainer.device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)

    checkpoints = [0, 1, 5, 10, 20]
    ckpt_results = {}

    print("\n============================================================")
    print("CRITIC ACTION LANDSCAPE DIAGNOSTIC EXPERIMENT")
    print("============================================================")

    # Initial evaluation at Step 0
    points_0, fd_0 = evaluate_action_landscape(critic, states_t, actions_t)
    ckpt_results[0] = (points_0, fd_0)

    for step in range(1, 21):
        with torch.no_grad():
            t_ed_actions = []
            for idx in range(trainer.num_ed):
                ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                t_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
            cloud_st = next_states_t[:, -11:]
            t_cloud_probs = trainer.cloud_target_actor(cloud_st)
            t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

            tQ = critic_trg(next_states_t, t_joint_action)
            ty = rewards_t + (1.0 - dones_t) * 0.99 * tQ

        cQ = critic(states_t, actions_t)
        loss = nn.MSELoss()(cQ, ty)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        soft_update(critic_trg, critic, 0.01)

        if step in checkpoints:
            points, fd = evaluate_action_landscape(critic, states_t, actions_t)
            ckpt_results[step] = (points, fd)

    for step_num in checkpoints:
        pts, fds = ckpt_results[step_num]
        print(f"\n------------------------------------------------------------")
        print(f"CHECKPOINT: STEP {step_num}")
        print(f"------------------------------------------------------------")
        print(f"{'Alpha':<8} | {'Q Mean':<10} | {'Q Min':<10} | {'Q Max':<10} | {'Q-Q(loc)':<10} | {'dQ/dA Diff':<12} | {'FD Slope':<10}")
        print("-" * 85)

        for i, p in enumerate(pts):
            fd_str = f"{fds[i]:<10.6f}" if i < len(fds) else "N/A"
            print(f"{p['alpha']:<8.2f} | {p['stats']['mean']:<10.6f} | {p['stats']['min']:<10.6f} | {p['stats']['max']:<10.6f} | {p['q_diff']:<+10.6f} | {p['dq_diff']:<+12.6f} | {fd_str}")

    print("\n============================================================")
    print("ACTION LANDSCAPE ANALYSIS & CLASSIFICATION")
    print("============================================================")

    pts_20, fds_20 = ckpt_results[20]

    # Analysis 1: Linearity check (compare mid-point Q with linear interpolation)
    q_loc = pts_20[0]['stats']['mean']
    q_off = pts_20[-1]['stats']['mean']
    q_mid_actual = pts_20[3]['stats']['mean']  # alpha = 0.5
    q_mid_linear = (q_loc + q_off) / 2.0
    linearity_err = abs(q_mid_actual - q_mid_linear)

    # Analysis 2: Gradient vs Finite Difference alignment check
    grad_fd_diffs = []
    for i in range(len(fds_20)):
        avg_grad = (pts_20[i]['dq_diff'] + pts_20[i+1]['dq_diff']) / 2.0
        grad_fd_diffs.append(abs(avg_grad - fds_20[i]))

    max_grad_fd_err = max(grad_fd_diffs)

    # Analysis 3: Curvature check (second derivative approximation)
    second_diffs = []
    for i in range(len(fds_20) - 1):
        second_diffs.append(abs(fds_20[i+1] - fds_20[i]))
    max_curvature = max(second_diffs) if second_diffs else 0.0

    print(f"Linearity error at alpha=0.5: {linearity_err:.6f}")
    print(f"Max Gradient vs Finite-Difference Error: {max_grad_fd_err:.6f}")
    print(f"Max Action-Surface Curvature: {max_curvature:.6f}")

    discrete_pref = (q_off > q_loc)
    grad_pref = (pts_20[0]['dq_diff'] > 0)

    if discrete_pref != grad_pref and abs(q_off - q_loc) > 1e-4:
        classification = "D. CRITIC GRADIENT IS MISALIGNED WITH DISCRETE ACTION VALUES"
    elif max_curvature > 2.0:
        classification = "B. CRITIC HAS EXCESSIVE ACTION CURVATURE"
    elif max_grad_fd_err > 0.5:
        classification = "C. CRITIC HAS DISCRETE/CONTINUOUS ACTION INCONSISTENCY"
    elif linearity_err < 0.1 and max_grad_fd_err < 0.2:
        classification = "A. CRITIC ACTION LANDSCAPE IS WELL-BEHAVED"
    else:
        classification = "E. NO ACTION-LANDSCAPE PROBLEM IDENTIFIED"

    print(f"\nFINAL CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
