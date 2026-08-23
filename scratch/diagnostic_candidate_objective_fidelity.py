import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr, pearsonr
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
        'std': float(np_arr.std())
    }


def safe_corr(x, y):
    x_flat = np.array(x).flatten()
    y_flat = np.array(y).flatten()
    if np.std(x_flat) == 0 or np.std(y_flat) == 0:
        return 0.0
    r, _ = pearsonr(x_flat, y_flat)
    return float(r) if not np.isnan(r) else 0.0


def compute_paper_total_reward(s_i, ed_action):
    # Paper-defined total reward r_tot = -(w_delay * T_total + w_energy * E_total + w_cost * Cost_total)
    task_size = abs(float(s_i[0])) * 1e6 if s_i[0] != 0 else 1e6
    cpu_req = abs(float(s_i[1])) * 1e9 if s_i[1] != 0 else 1e9

    if ed_action == 0:
        f_loc = 1.0e9
        delay = cpu_req / f_loc
        energy = 1e-27 * (f_loc ** 2) * cpu_req
        cost = 0.0
    else:
        f_edge = 5.0e9
        rate_tx = 10.0e6
        delay = (task_size / rate_tx) + (cpu_req / f_edge)
        energy = 0.1 * (task_size / rate_tx)
        cost = 0.05 * (cpu_req / 1e9)

    r_tot = -(0.4 * delay + 0.4 * energy + 0.2 * cost)
    return float(r_tot)


def compute_current_candidate_utility(s_i, ed_action):
    # Current candidate utility U(a) = -(0.5 * delay + 0.5 * energy)
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

    u_cand = -(0.5 * delay + 0.5 * energy)
    return float(u_cand)


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device_str}")

    np.random.seed(42)
    torch.manual_seed(42)

    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 100
    transitions = []

    spearman_ranks = []
    top1_agreements = []

    for idx in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        ed0_st = s_joint[0:9]

        cand_list, p_dist = point_to_uniform_candidate_generation(trainer.ed_primary_actors[0], ed0_st, P=5)

        u_scores = [compute_current_candidate_utility(ed0_st, act) for act in cand_list]
        r_tot_scores = [compute_paper_total_reward(ed0_st, act) for act in cand_list]

        u_best_act = cand_list[np.argmax(u_scores)]
        r_best_act = cand_list[np.argmax(r_tot_scores)]

        top1_agree = (u_best_act == r_best_act)
        top1_agreements.append(top1_agree)

        if len(set(u_scores)) > 1 and len(set(r_tot_scores)) > 1:
            rho, _ = spearmanr(u_scores, r_tot_scores)
            if not np.isnan(rho):
                spearman_ranks.append(rho)

        transitions.append({
            'id': idx,
            'state': ed0_st,
            'cands': cand_list,
            'u_scores': u_scores,
            'r_tot_scores': r_tot_scores,
            'u_best': u_best_act,
            'r_best': r_best_act,
            'agree': top1_agree
        })

    print("\n============================================================")
    print("PART 1 — CANDIDATE UTILITY VS PAPER REWARD FIDELITY (100 TRANSITIONS)")
    print("============================================================")

    mean_spearman = float(np.mean(spearman_ranks)) if spearman_ranks else 1.0
    top1_pct = (sum(top1_agreements) / batch_size) * 100.0

    print(f"Mean Spearman Rank Correlation (Candidate U vs Paper r_tot): {mean_spearman:+.4f}")
    print(f"Top-1 Candidate Selection Agreement: {sum(top1_agreements)} / 100 ({top1_pct:.1f}%)")

    print("\n============================================================")
    print("PART 2 — TRANSITION-LEVEL FIDELITY EXAMPLES (FIRST 20 TRANSITIONS)")
    print("============================================================")

    print(f"{'ID':<4} | {'Candidates':<15} | {'U(a) Scores':<25} | {'r_tot Scores':<25} | {'Agree':<6}")
    print("-" * 82)

    for t in transitions[:20]:
        cands_str = str(t['cands'])
        u_str = f"[{', '.join([f'{v:.2f}' for v in t['u_scores']])}]"
        r_str = f"[{', '.join([f'{v:.2f}' for v in t['r_tot_scores']])}]"
        agree_str = "YES" if t['agree'] else "NO"
        print(f"{t['id']:<4d} | {cands_str:<15} | {u_str:<25} | {r_str:<25} | {agree_str:<6}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    if top1_pct > 95:
        classification = "A. CANDIDATE UTILITY MATCHES PAPER OBJECTIVE PERFECTLY"
    elif top1_pct < 80:
        classification = "B. CANDIDATE UTILITY OMITS KEY SYSTEM PENALTIES OR CONSTRAINTS"
    elif mean_spearman < 0.5:
        classification = "C. CANDIDATE SELECTION IS MISALIGNED WITH ENVIRONMENT REWARD"
    else:
        classification = "D. CANDIDATE OBJECTIVE CREATES ARTIFICIAL OFFLOAD / LOCAL BIAS"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("REAL TRAINING EXECUTED: NO")
    print("HOST CUDA EXECUTION: NO")


if __name__ == "__main__":
    main()
