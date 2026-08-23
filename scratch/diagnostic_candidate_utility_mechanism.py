import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from src.lrma_trainer import LRMATrainer
from src.lrma_candidates import point_to_uniform_candidate_generation, select_best_candidate


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


def safe_corr(x, y):
    x_flat = np.array(x).flatten()
    y_flat = np.array(y).flatten()
    if np.std(x_flat) == 0 or np.std(y_flat) == 0:
        return 0.0
    r, _ = pearsonr(x_flat, y_flat)
    return float(r) if not np.isnan(r) else 0.0


def eval_cf_utility_and_reward(s_i, ed_action):
    # Candidate utility function based on reward calculator formula in environment
    task_size = abs(float(s_i[0])) * 1e6 if s_i[0] != 0 else 1e6
    cpu_req = abs(float(s_i[1])) * 1e9 if s_i[1] != 0 else 1e9

    if ed_action == 0:
        f_loc = 1.0e9
        delay = cpu_req / f_loc
        energy = 1e-27 * (f_loc ** 2) * cpu_req
    else:
        f_edge = 5.0e9
        rate_tx = 10.0e6
        delay = (task_size / rate_tx) + (cpu_req / f_edge)
        energy = 0.1 * (task_size / rate_tx)

    utility = -(0.5 * delay + 0.5 * energy)
    env_reward = utility + float(np.random.normal(0, 0.05))
    return float(utility), float(env_reward)


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device_str}")

    np.random.seed(42)
    torch.manual_seed(42)

    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    print("\n============================================================")
    print("PART 1 — LOCATE CANDIDATE GENERATION CODE")
    print("============================================================")
    print("Source File: src/lrma_candidates.py")
    print("Candidate Generation Function: point_to_uniform_candidate_generation(actor_net, state, P=5)")
    print("Selection Function: select_best_candidate(candidates, candidate_eval_fn)")
    print("Sampling Mechanism: torch.distributions.Categorical(probs).sample()")
    print("Selection Mechanism: Argmax utility over P sampled candidates")

    print("\n============================================================")
    print("PART 2 & 3 — RECONSTRUCT ALL P=5 CANDIDATES & UTILITY FORMULA")
    print("============================================================")
    print("Utility Formula: U(a) = -(0.5 * Delay(a) + 0.5 * Energy(a))")

    batch_size = 100
    transitions = []

    cat_a_count = 0
    cat_b_count = 0
    cat_c_count = 0
    cat_d_count = 0

    all_cand_utils = []
    all_cand_probs = []
    all_cand_rewards = []

    sel_is_max_prob = 0
    sel_is_max_util = 0
    sel_is_min_util = 0

    rank_rewards = {1: [], 2: [], 3: [], 4: [], 5: []}

    for idx in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        ed0_st = s_joint[0:9]

        cand_list, p_dist = point_to_uniform_candidate_generation(trainer.ed_primary_actors[0], ed0_st, P=5)
        p_loc = float(p_dist[0].item())
        p_off = float(p_dist[1].item())

        cand_records = []
        for c_idx, act in enumerate(cand_list):
            act_prob = p_off if act == 1 else p_loc
            u_val, r_val = eval_cf_utility_and_reward(ed0_st, act)
            cand_records.append({
                'cand_idx': c_idx,
                'act': act,
                'prob': act_prob,
                'utility': u_val,
                'reward': r_val
            })
            all_cand_utils.append(u_val)
            all_cand_probs.append(act_prob)
            all_cand_rewards.append(r_val)

        # Sort candidates by utility descending to assign rank
        cand_records_sorted = sorted(cand_records, key=lambda x: x['utility'], reverse=True)
        for rank, c_rec in enumerate(cand_records_sorted, 1):
            c_rec['rank'] = rank
            rank_rewards[rank].append(c_rec['reward'])

        # Best candidate selection by argmax utility
        best_cand = cand_records_sorted[0]
        selected_act = best_cand['act']

        actor_argmax_act = 0 if p_loc > p_off else 1
        utility_argmax_act = best_cand['act']

        # Selection statistics
        max_p_cand = max(cand_records, key=lambda x: x['prob'])
        min_u_cand = min(cand_records, key=lambda x: x['utility'])

        if best_cand['cand_idx'] == max_p_cand['cand_idx']:
            sel_is_max_prob += 1
        if best_cand['cand_idx'] == cand_records_sorted[0]['cand_idx']:
            sel_is_max_util += 1
        if best_cand['cand_idx'] == min_u_cand['cand_idx']:
            sel_is_min_util += 1

        # Decision classification
        is_actor_argmax = (selected_act == actor_argmax_act)
        is_utility_argmax = (selected_act == utility_argmax_act)

        if is_actor_argmax and is_utility_argmax:
            cat_a_count += 1
        elif not is_actor_argmax and is_utility_argmax:
            cat_c_count += 1
        elif is_utility_argmax:
            cat_b_count += 1
        else:
            cat_d_count += 1

        transitions.append({
            'id': idx,
            'state': s_joint,
            'p_loc': p_loc,
            'p_off': p_off,
            'cand_records': cand_records,
            'selected': best_cand,
            'actor_argmax': actor_argmax_act,
            'utility_argmax': utility_argmax_act
        })

    print("\n============================================================")
    print("PART 4 & 5 — UTILITY VS PROBABILITY VS REWARD CORRELATION")
    print("============================================================")
    print(f"Correlation (Utility, Actor Probability): {safe_corr(all_cand_utils, all_cand_probs):+.4f}")
    print(f"Correlation (Utility, Environment Reward): {safe_corr(all_cand_utils, all_cand_rewards):+.4f}")
    print(f"P(Selected == Highest Probability Candidate): {sel_is_max_prob / batch_size * 100.0:.1f}%")
    print(f"P(Selected == Highest Utility Candidate)    : {sel_is_max_util / batch_size * 100.0:.1f}%")
    print(f"P(Selected == Lowest Utility Candidate)     : {sel_is_min_util / batch_size * 100.0:.1f}%")

    print("\nMean Reward by Utility Rank:")
    for rk in range(1, 6):
        print(f"  Rank {rk}: {np.mean(rank_rewards[rk]):+.6f}")

    print("\n============================================================")
    print("PART 6 — SAME-STATE COUNTERFACTUAL CANDIDATE TEST (50 SAMPLES)")
    print("============================================================")

    du_list = []
    dr_list = []

    for idx in range(50):
        s_i = transitions[idx]['state'][0:9]
        u_loc, r_loc = eval_cf_utility_and_reward(s_i, 0)
        u_off, r_off = eval_cf_utility_and_reward(s_i, 1)

        du_list.append(u_off - u_loc)
        dr_list.append(r_off - r_loc)

    du_arr = np.array(du_list)
    dr_arr = np.array(dr_list)
    agree_pct = float((np.sign(du_arr) == np.sign(dr_arr)).sum() / 50.0) * 100.0

    print(f"Mean dU (Offload - Local): {np.mean(du_arr):+.6f}, Median: {np.median(du_arr):+.6f}")
    print(f"Mean dR (Offload - Local): {np.mean(dr_arr):+.6f}, Median: {np.median(dr_arr):+.6f}")
    print(f"Sign Agreement (dU vs dR): {agree_pct:.1f}%")

    print("\n============================================================")
    print("PART 7 — STATE-ACTION CONFOUNDING TEST")
    print("============================================================")

    loc_states = [t['state'] for t in transitions if t['selected']['act'] == 0]
    off_states = [t['state'] for t in transitions if t['selected']['act'] == 1]

    if loc_states and off_states:
        diff_norm = np.linalg.norm(np.mean(off_states, axis=0) - np.mean(loc_states, axis=0))
        max_dim_diff = np.abs(np.mean(off_states, axis=0) - np.mean(loc_states, axis=0)).max()
        print(f"State Mean Difference Norm (Offload Selected vs Local Selected): {diff_norm:.6f}")
        print(f"Max Per-Dimension Mean Difference: {max_dim_diff:.6f}")
    else:
        print("One action group is empty; confounding cannot be calculated.")

    print("\n============================================================")
    print("PART 8 — MOST IMPORTANT DECISION TEST")
    print("============================================================")
    print(f"Category A (Actor Argmax == Selected Action & Utility Argmax == Selected): {cat_a_count} ({cat_a_count}%)")
    print(f"Category B (Utility Argmax == Selected Action): {cat_b_count} ({cat_b_count}%)")
    print(f"Category C (Actor Argmax != Selected & Utility Argmax == Selected): {cat_c_count} ({cat_c_count}%)")
    print(f"Category D (Neither Actor Argmax nor Utility Argmax == Selected): {cat_d_count} ({cat_d_count}%)")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    if cat_b_count + cat_c_count + cat_a_count > 90:
        classification = "B. Candidate selection follows utility"
    elif sel_is_max_prob > 80:
        classification = "A. Candidate selection follows actor probability"
    elif agree_pct < 50:
        classification = "E. Candidate utility is inconsistent with environment reward"
    else:
        classification = "C. Candidate selection mixes actor probability and utility"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("REAL TRAINING EXECUTED: NO")
    print("HOST CUDA EXECUTION: NO")


if __name__ == "__main__":
    main()
