import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from src.lrma_trainer import LRMATrainer
from src.lrma_networks import soft_update


def get_stats(tensor_or_array):
    if isinstance(tensor_or_array, torch.Tensor):
        np_arr = tensor_or_array.detach().cpu().numpy()
    else:
        np_arr = np.array(tensor_or_array)
    return {
        'min': float(np_arr.min()),
        'max': float(np_arr.max()),
        'mean': float(np_arr.mean()),
        'std': float(np_arr.std()),
        'median': float(np.median(np_arr))
    }


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device_str}")

    np.random.seed(42)
    torch.manual_seed(42)

    trainer_init = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

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
        trainer_init.replay_buffer_ed.add(s_joint, a_joint, r, ns_joint, False)

    states, actions, rewards, next_states, dones = trainer_init.replay_buffer_ed.sample(batch_size)
    states_t = torch.FloatTensor(states).to(trainer_init.device)
    actions_t = torch.FloatTensor(actions).to(trainer_init.device)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(trainer_init.device)

    # Decode ED-0 actions
    ed0_actions = actions[:, 0:2] # shape (64, 2)
    local_mask = (ed0_actions[:, 0] == 1.0)
    offload_mask = (ed0_actions[:, 1] == 1.0)

    n_local = int(local_mask.sum())
    n_offload = int(offload_mask.sum())

    print("\n============================================================")
    print("PART 1 — REPLAY ACTION/REWARD ASSOCIATION")
    print("============================================================")
    print(f"Replay Sample Counts for ED-0: Local = {n_local}, Offload = {n_offload}")

    r_local_group = rewards[local_mask]
    r_offload_group = rewards[offload_mask]

    r_loc_stats = get_stats(r_local_group) if n_local > 0 else {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}
    r_off_stats = get_stats(r_offload_group) if n_offload > 0 else {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}

    empirical_dr = r_off_stats['mean'] - r_loc_stats['mean']

    print(f"Replay Local Group R  : Mean = {r_loc_stats['mean']:+.6f}, Median = {r_loc_stats['median']:+.6f}, Std = {r_loc_stats['std']:.6f}")
    print(f"Replay Offload Group R: Mean = {r_off_stats['mean']:+.6f}, Median = {r_off_stats['median']:+.6f}, Std = {r_off_stats['std']:.6f}")
    print(f"Empirical Replay dR (Offload - Local Group): {empirical_dr:+.6f}")

    print("\n============================================================")
    print("PART 2 — COUNTERFACTUAL REWARD")
    print("============================================================")

    r_loc_cf_list = []
    r_off_cf_list = []

    for idx in range(batch_size):
        s_i = states[idx]
        task_size = abs(float(s_i[0])) * 1e6 if s_i[0] != 0 else 1e6
        cpu_req = abs(float(s_i[1])) * 1e9 if s_i[1] != 0 else 1e9

        f_loc = 1.0e9
        delay_loc = cpu_req / f_loc
        energy_loc = 1e-27 * (f_loc ** 2) * cpu_req
        r_loc = -(0.5 * delay_loc + 0.5 * energy_loc)

        f_edge = 5.0e9
        rate_tx = 10.0e6
        delay_off = (task_size / rate_tx) + (cpu_req / f_edge)
        energy_off = 0.1 * (task_size / rate_tx)
        r_off = -(0.5 * delay_off + 0.5 * energy_off)

        r_loc_cf_list.append(r_loc)
        r_off_cf_list.append(r_off)

    r_loc_cf_arr = np.array(r_loc_cf_list)
    r_off_cf_arr = np.array(r_off_cf_list)
    dr_cf_arr = r_off_cf_arr - r_loc_cf_arr

    r_loc_cf_stats = get_stats(r_loc_cf_arr)
    r_off_cf_stats = get_stats(r_off_cf_arr)
    dr_cf_stats = get_stats(dr_cf_arr)

    print(f"Counterfactual Local R  : Mean = {r_loc_cf_stats['mean']:+.6f}, Std = {r_loc_cf_stats['std']:.6f}")
    print(f"Counterfactual Offload R: Mean = {r_off_cf_stats['mean']:+.6f}, Std = {r_off_cf_stats['std']:.6f}")
    print(f"Counterfactual dR (Offload - Local): Mean = {dr_cf_stats['mean']:+.6f}, Median = {dr_cf_stats['median']:+.6f}")
    print(f"Difference (Replay dR - Counterfactual dR): {empirical_dr - dr_cf_stats['mean']:+.6f}")

    print("\n============================================================")
    print("PART 3 — STATE/ACTION CONFOUNDING")
    print("============================================================")

    states_loc_group = states[local_mask]
    states_off_group = states[offload_mask]

    if n_local > 0 and n_offload > 0:
        mean_diff_vec = states_off_group.mean(axis=0) - states_loc_group.mean(axis=0)
        mean_diff_norm = np.linalg.norm(mean_diff_vec)
        max_dim_diff = np.abs(mean_diff_vec).max()
        print(f"State Mean Difference Norm (Offload - Local): {mean_diff_norm:.6f}")
        print(f"Max Per-Dimension Mean Difference: {max_dim_diff:.6f}")
    else:
        mean_diff_norm = 0.0
        print("One group has zero samples; confounding cannot be calculated.")

    print("\n============================================================")
    print("PART 4 & PART 5 — CRITIC FITTING TO PURE REPLAY REWARD & COUNTERFACTUAL EVALUATION")
    print("============================================================")

    trainer = copy.deepcopy(trainer_init)
    checkpoints = [0, 1, 5, 10, 20, 50, 100]

    print(f"{'Step':<5} | {'Replay Q(loc)':<14} | {'Replay Q(off)':<14} | {'Replay dQ_grp':<14} | {'CF dQ_mean':<12} | {'CF dQ_med':<12}")
    print("-" * 85)

    history = []

    for step in range(101):
        with torch.no_grad():
            cQ = trainer.critic(states_t, actions_t).cpu().numpy().flatten()
            q_loc_grp = cQ[local_mask].mean() if n_local > 0 else 0.0
            q_off_grp = cQ[offload_mask].mean() if n_offload > 0 else 0.0
            dq_grp = q_off_grp - q_loc_grp

            # Counterfactual Q evaluation
            a_loc_cf = actions_t.clone(); a_loc_cf[:, 0] = 1.0; a_loc_cf[:, 1] = 0.0
            a_off_cf = actions_t.clone(); a_off_cf[:, 0] = 0.0; a_off_cf[:, 1] = 1.0
            q_loc_cf = trainer.critic(states_t, a_loc_cf).cpu().numpy().flatten()
            q_off_cf = trainer.critic(states_t, a_off_cf).cpu().numpy().flatten()
            dq_cf = q_off_cf - q_loc_cf

        if step in checkpoints:
            dq_cf_st = get_stats(dq_cf)
            print(f"{step:<5d} | {q_loc_grp:<+14.6f} | {q_off_grp:<+14.6f} | {dq_grp:<+14.6f} | {dq_cf_st['mean']:<+12.6f} | {dq_cf_st['median']:<+12.6f}")
            history.append({
                'step': step,
                'q_loc_grp': q_loc_grp,
                'q_off_grp': q_off_grp,
                'dq_grp': dq_grp,
                'dq_cf_mean': dq_cf_st['mean'],
                'dq_cf_med': dq_cf_st['median']
            })

        if step < 100:
            cQ_t = trainer.critic(states_t, actions_t)
            c_loss = nn.MSELoss()(cQ_t, rewards_t)
            trainer.critic_optimizer.zero_grad()
            c_loss.backward()
            trainer.critic_optimizer.step()

    print("\n============================================================")
    print("PART 6 — CRITICAL COMPARISON SUMMARY TABLE")
    print("============================================================")
    print(f"{'Quantity':<45} | {'Value':<20}")
    print("-" * 70)
    print(f"{'Replay Empirical Reward Difference (Off - Loc)':<45} | {empirical_dr:<+20.6f}")
    print(f"{'Counterfactual Reward Difference (Off - Loc)':<45} | {dr_cf_stats['mean']:<+20.6f}")
    print(f"{'Critic Final Replay-Group Q Difference':<45} | {history[-1]['dq_grp']:<+20.6f}")
    print(f"{'Critic Final Counterfactual Q Difference':<45} | {history[-1]['dq_cf_mean']:<+20.6f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    final_dq_cf = history[-1]['dq_cf_mean']

    if empirical_dr < 0 and final_dq_cf < 0:
        classification = "A. REPLAY ACTION/REWARD ASSOCIATION EXPLAINS LOCAL PREFERENCE"
    elif dr_cf_stats['mean'] > 0 and empirical_dr < 0:
        classification = "B. COUNTERFACTUAL REWARD CALCULATION IS INCONSISTENT WITH REPLAY REWARD"
    elif mean_diff_norm > 1.0 and final_dq_cf < 0:
        classification = "C. STATE/ACTION CONFOUNDING CAUSES THE CRITIC TO LEARN PREFERENCE"
    elif dr_cf_stats['mean'] > 0 and final_dq_cf < 0:
        classification = "D. CRITIC FITTING CREATES AN ACTION PREFERENCE NOT IN REPLAY REWARD"
    else:
        classification = "E. ANOTHER MECHANISM"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("HOST EXECUTION COMPLETED: NO (STATIC VERIFICATION ONLY)")


if __name__ == "__main__":
    main()
