import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from src.lrma_trainer import LRMATrainer


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
        'abs_mean': float(np.abs(np_arr).mean())
    }


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
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(trainer.device)
    next_states_t = torch.FloatTensor(next_states).to(trainer.device)
    dones_t = torch.FloatTensor(dones).unsqueeze(1).to(trainer.device)

    print("\n============================================================")
    print("PART 1: REPLAY STATE-ACTION DIVERSITY")
    print("============================================================")
    print(f"states shape     : {states.shape}")
    print(f"actions shape    : {actions.shape}")
    print(f"rewards shape    : {rewards.shape}")
    print(f"next_states shape: {next_states.shape}")
    print(f"dones shape      : {dones.shape}")

    decoded_ed_actions = []
    ed_counts = []
    for idx in range(25):
        oh_chunk = actions[:, idx*2 : (idx+1)*2]
        act_indices = np.argmax(oh_chunk, axis=1)
        decoded_ed_actions.append(act_indices)
        loc_c = int((act_indices == 0).sum())
        off_c = int((act_indices == 1).sum())
        ed_counts.append({
            'ed_idx': idx,
            'local': loc_c,
            'offload': off_c,
            'loc_frac': loc_c / batch_size,
            'off_frac': off_c / batch_size
        })

    cloud_act_indices = np.argmax(actions[:, 50:55], axis=1)
    cloud_counts = [int((cloud_act_indices == m).sum()) for m in range(5)]

    print(f"\nED-0 Action Counts: Local = {ed_counts[0]['local']} ({ed_counts[0]['loc_frac']:.2f}), Offload = {ed_counts[0]['offload']} ({ed_counts[0]['off_frac']:.2f})")
    print(f"Cloud MES Action Counts (MES 0..4): {cloud_counts}")

    print("\n============================================================")
    print("PART 2: UNIQUE JOINT ACTIONS & HAMMING DISTANCE")
    print("============================================================")

    unique_actions = np.unique(actions, axis=0)
    num_unique = len(unique_actions)
    unique_frac = num_unique / batch_size

    hamming_dists = []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            h_dist = np.sum(actions[i] != actions[j])
            hamming_dists.append(h_dist)

    h_stats = get_stats(hamming_dists)
    print(f"Unique Joint Actions: {num_unique}/{batch_size} ({unique_frac*100:.1f}%)")
    print(f"Pairwise Hamming Distance: Mean = {h_stats['mean']:.2f}, Min = {h_stats['min']:.0f}, Max = {h_stats['max']:.0f}")

    print("\n============================================================")
    print("PART 3: STATE DIVERSITY")
    print("============================================================")

    state_dists = []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            d = np.linalg.norm(states[i] - states[j])
            state_dists.append(d)

    s_dist_stats = get_stats(state_dists)
    dim_vars = np.var(states, axis=0)
    near_zero_var_dims = int((dim_vars < 1e-5).sum())

    print(f"Pairwise State Distance: Mean = {s_dist_stats['mean']:.4f}, Min = {s_dist_stats['min']:.4f}, Max = {s_dist_stats['max']:.4f}")
    print(f"State Dimension Variance: Min = {dim_vars.min():.6f}, Max = {dim_vars.max():.6f}")
    print(f"Near-Zero Variance State Dimensions (< 1e-5): {near_zero_var_dims}/{states.shape[1]}")

    print("\n============================================================")
    print("PART 4: ED-0 COUNTERFACTUAL COVERAGE")
    print("============================================================")

    a_loc = actions_t.clone()
    a_loc[:, 0] = 1.0
    a_loc[:, 1] = 0.0

    a_off = actions_t.clone()
    a_off[:, 0] = 0.0
    a_off[:, 1] = 1.0

    with torch.no_grad():
        q_loc = trainer.critic(states_t, a_loc)
        q_off = trainer.critic(states_t, a_off)

    delta_q = (q_off - q_loc).cpu().numpy().flatten()
    dq_stats = get_stats(delta_q)
    median_dq = float(np.median(delta_q))

    print(f"Counterfactual Q(offload) - Q(local) for ED-0:")
    print(f"  Mean   : {dq_stats['mean']:+.6f}")
    print(f"  Median : {median_dq:+.6f}")
    print(f"  Std    : {dq_stats['std']:.6f}")
    print(f"  Min    : {dq_stats['min']:+.6f}")
    print(f"  Max    : {dq_stats['max']:+.6f}")

    print("\n============================================================")
    print("PART 5: CRITIC FIT TO OBSERVED TRANSITIONS")
    print("============================================================")

    with torch.no_grad():
        q_replay = trainer.critic(states_t, actions_t).cpu().numpy().flatten()

        t_ed_actions = []
        for idx in range(trainer.num_ed):
            ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
            t_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
        cloud_st = next_states_t[:, -11:]
        t_cloud_probs = trainer.cloud_target_actor(cloud_st)
        t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

        tQ = trainer.critic_target(next_states_t, t_joint_action)
        ty = (rewards_t + (1.0 - dones_t) * 0.99 * tQ).cpu().numpy().flatten()

    mse_qr = float(np.mean((q_replay - rewards) ** 2))
    mae_qr = float(np.mean(np.abs(q_replay - rewards)))
    corr_qr, _ = pearsonr(q_replay, rewards)

    mse_qy = float(np.mean((q_replay - ty) ** 2))
    mae_qy = float(np.mean(np.abs(q_replay - ty)))
    corr_qy, _ = pearsonr(q_replay, ty)

    print(f"Critic Q vs Observed Reward R:")
    print(f"  MSE(Q, R) : {mse_qr:.6f} | MAE(Q, R) : {mae_qr:.6f} | Correlation(Q, R) : {corr_qr:+.6f}")
    print(f"Critic Q vs Target Y:")
    print(f"  MSE(Q, Y) : {mse_qy:.6f} | MAE(Q, Y) : {mae_qy:.6f} | Correlation(Q, Y) : {corr_qy:+.6f}")

    print("\n============================================================")
    print("PART 6: ACTION-REWARD IDENTIFIABILITY (ED-0 LOCAL VS OFFLOAD)")
    print("============================================================")

    ed0_acts = decoded_ed_actions[0]
    loc_mask = (ed0_acts == 0)
    off_mask = (ed0_acts == 1)

    r_loc_grp = rewards[loc_mask]
    r_off_grp = rewards[off_mask]

    q_loc_grp = q_replay[loc_mask]
    q_off_grp = q_replay[off_mask]

    print(f"ED-0 Local Group   (N={len(r_loc_grp)}): Mean R = {r_loc_grp.mean():+.4f} (std {r_loc_grp.std():.4f}), Mean Q = {q_loc_grp.mean():+.4f} (std {q_loc_grp.std():.4f})")
    print(f"ED-0 Offload Group (N={len(r_off_grp)}): Mean R = {r_off_grp.mean():+.4f} (std {r_off_grp.std():.4f}), Mean Q = {q_off_grp.mean():+.4f} (std {q_off_grp.std():.4f})")
    print(f"Group Delta R (Offload - Local): {r_off_grp.mean() - r_loc_grp.mean():+.6f}")
    print(f"Group Delta Q (Offload - Local): {q_off_grp.mean() - q_loc_grp.mean():+.6f}")

    print("\n============================================================")
    print("PART 7: CRITIC SENSITIVITY PER ED AGENT")
    print("============================================================")

    sens_results = []
    for ed_idx in range(25):
        a_pert = actions_t.clone()
        a_pert[:, ed_idx*2] = 1.0 - actions_t[:, ed_idx*2]
        a_pert[:, ed_idx*2+1] = 1.0 - actions_t[:, ed_idx*2+1]

        with torch.no_grad():
            q_pert = trainer.critic(states_t, a_pert).cpu().numpy().flatten()

        dq = q_pert - q_replay
        abs_dq = np.abs(dq)
        sens_results.append({
            'ed_idx': ed_idx,
            'mean_abs_dq': float(abs_dq.mean()),
            'mean_dq': float(dq.mean()),
            'std_dq': float(dq.std())
        })

    print(f"{'ED Index':<10} | {'Mean |delta_Q|':<15} | {'Mean delta_Q':<15} | {'Std delta_Q':<15}")
    print("-" * 60)
    for sr in sens_results[:5]:
        print(f"{sr['ed_idx']:<10d} | {sr['mean_abs_dq']:<15.6f} | {sr['mean_dq']:<+15.6f} | {sr['std_dq']:<15.6f}")

    avg_sens_all = float(np.mean([sr['mean_abs_dq'] for sr in sens_results]))
    print(f"... (Average Mean |delta_Q| across all 25 EDs: {avg_sens_all:.6f})")

    print("\n============================================================")
    print("PART 8: CRITIC VS REWARD SCALE COMPARISON")
    print("============================================================")

    r_st = get_stats(rewards)
    q_st = get_stats(q_replay)
    y_st = get_stats(ty)

    print(f"Reward   : Mean = {r_st['mean']:+.4f}, Std = {r_st['std']:.4f}, |Mean| = {r_st['abs_mean']:.4f}, Range = [{r_st['min']:.4f}, {r_st['max']:.4f}]")
    print(f"Critic Q : Mean = {q_st['mean']:+.4f}, Std = {q_st['std']:.4f}, |Mean| = {q_st['abs_mean']:.4f}, Range = [{q_st['min']:.4f}, {q_st['max']:.4f}]")
    print(f"Target Y : Mean = {y_st['mean']:+.4f}, Std = {y_st['std']:.4f}, |Mean| = {y_st['abs_mean']:.4f}, Range = [{y_st['min']:.4f}, {y_st['max']:.4f}]")

    print("\n============================================================")
    print("PART 9: IN-MEMORY FRESH CRITIC FITTING (20 STEPS)")
    print("============================================================")

    fresh_critic = copy.deepcopy(trainer.critic).to(trainer.device)
    fresh_opt = torch.optim.Adam(fresh_critic.parameters(), lr=0.001)

    print(f"{'Step':<6} | {'MSE Loss':<12} | {'Q Mean':<12} | {'Q Std':<12} | {'Corr(Q, Y)':<12}")
    print("-" * 60)

    for step in range(1, 21):
        cQ_f = fresh_critic(states_t, actions_t)
        loss_f = nn.MSELoss()(cQ_f, torch.FloatTensor(ty).unsqueeze(1).to(trainer.device))

        fresh_opt.zero_grad()
        loss_f.backward()
        fresh_opt.step()

        if step % 5 == 0 or step == 1:
            with torch.no_grad():
                q_f_np = fresh_critic(states_t, actions_t).cpu().numpy().flatten()
                c_qy, _ = pearsonr(q_f_np, ty)
                print(f"{step:<6d} | {loss_f.item():<12.6f} | {q_f_np.mean():<+12.6f} | {q_f_np.std():<12.6f} | {c_qy:<+12.6f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    severe_imbalance = any(r['loc_frac'] < 0.1 or r['off_frac'] < 0.1 for r in ed_counts)
    low_diversity = (unique_frac < 0.5)
    low_sensitivity = (avg_sens_all < 1e-4)

    if low_sensitivity:
        classification = "D. CRITIC HAS LOW ACTION SENSITIVITY"
    elif severe_imbalance:
        classification = "B. REPLAY HAS SEVERE ACTION IMBALANCE"
    elif low_diversity:
        classification = "C. REPLAY HAS INSUFFICIENT JOINT-ACTION DIVERSITY"
    elif avg_sens_all > 1e-3 and num_unique == batch_size:
        classification = "A. REPLAY COVERAGE IS SUFFICIENT"
    else:
        classification = "G. INCONCLUSIVE"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
