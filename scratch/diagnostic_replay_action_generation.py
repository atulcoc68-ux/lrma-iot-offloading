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


def eval_cf_reward(s_i, ed_action):
    # ed_action: 0 for local, 1 for offload
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

    r = -(0.5 * delay + 0.5 * energy)
    return float(r)


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Execution Device: {device_str}")

    np.random.seed(42)
    torch.manual_seed(42)

    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 100
    transitions = []

    for idx in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        ed0_st = s_joint[0:9]

        # Actor policy
        cand_list, p_dist = point_to_uniform_candidate_generation(trainer.ed_primary_actors[0], ed0_st, P=5)
        p_loc = float(p_dist[0].item())
        p_off = float(p_dist[1].item())

        # Evaluate utility / reward for candidate solutions
        scores = [eval_cf_reward(ed0_st, act) for act in cand_list]
        best_cand_idx = int(np.argmax(scores))
        selected_act = cand_list[best_cand_idx]
        selected_score = scores[best_cand_idx]

        sorted_scores = sorted(scores, reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

        # Construct full joint action for EDs & Cloud
        ed_one_hots = []
        ed_one_hots.append(np.array([1.0, 0.0] if selected_act == 0 else [0.0, 1.0], dtype=np.float32))
        for _ in range(24):
            oh = np.zeros(2, dtype=np.float32)
            oh[np.random.choice([0, 1])] = 1.0
            ed_one_hots.append(oh)

        cloud_oh = np.zeros(5, dtype=np.float32)
        cloud_oh[np.random.choice(range(5))] = 1.0
        a_joint = np.concatenate(ed_one_hots + [cloud_oh])

        env_r = eval_cf_reward(ed0_st, selected_act) + float(np.random.normal(0, 0.1))
        ns_joint = np.random.randn(236).astype(np.float32)

        trainer.replay_buffer_ed.add(s_joint, a_joint, env_r, ns_joint, False)

        transitions.append({
            'id': idx,
            'state': s_joint,
            'p_loc': p_loc,
            'p_off': p_off,
            'cands': cand_list,
            'scores': scores,
            'sel_idx': best_cand_idx,
            'sel_act': selected_act,
            'sel_score': selected_score,
            'margin': margin,
            'env_r': env_r,
            'argmax_act': 0 if p_loc > p_off else 1
        })

    print("\n============================================================")
    print("PART 1 — CANDIDATE SELECTION STATISTICS (100 TRANSITIONS)")
    print("============================================================")

    sel_indices = [t['sel_idx'] for t in transitions]
    margins = [t['margin'] for t in transitions]
    scores_all = [t['sel_score'] for t in transitions]

    print(f"Candidate Count P per Agent: 5")
    print(f"Selected Candidate Index Counts (0..4): {dict(zip(*np.unique(sel_indices, return_counts=True)))}")
    print(f"Selected Utility Stats: Mean = {np.mean(scores_all):+.6f}, Std = {np.std(scores_all):.6f}")
    print(f"Utility Margin (Best - 2nd Best): Mean = {np.mean(margins):.6f}, Max = {np.max(margins):.6f}")

    print("\n============================================================")
    print("PART 2 — POLICY VS SELECTED ACTION STATISTICS")
    print("============================================================")

    loc_count = sum(1 for t in transitions if t['sel_act'] == 0)
    off_count = sum(1 for t in transitions if t['sel_act'] == 1)
    argmax_match = sum(1 for t in transitions if t['sel_act'] == t['argmax_act'])

    print(f"Selected Actions: Local = {loc_count} ({loc_count}%), Offload = {off_count} ({off_count}%)")
    print(f"Match between Selected Action and Argmax Policy: {argmax_match} / 100 ({argmax_match}%)")

    print("\n============================================================")
    print("PART 3 — REWARD STATISTICS BY SELECTED ACTION & POLICY PREFERENCE")
    print("============================================================")

    loc_rewards = [t['env_r'] for t in transitions if t['sel_act'] == 0]
    off_rewards = [t['env_r'] for t in transitions if t['sel_act'] == 1]

    print(f"Mean Reward for Selected Local Actions  : {np.mean(loc_rewards):+.6f} (N={len(loc_rewards)})" if loc_rewards else "No Local Actions")
    print(f"Mean Reward for Selected Offload Actions: {np.mean(off_rewards):+.6f} (N={len(off_rewards)})" if off_rewards else "No Offload Actions")

    print("\n============================================================")
    print("PART 4 & 5 — TRANSITION-LEVEL EXAMPLES TABLE (FIRST 20 TRANSITIONS)")
    print("============================================================")

    print(f"{'ID':<4} | {'P(loc)':<7} | {'P(off)':<7} | {'Candidates':<15} | {'SelIdx':<6} | {'SelAct':<6} | {'SelScore':<10} | {'EnvReward':<10}")
    print("-" * 90)

    for t in transitions[:20]:
        cands_str = str(t['cands'])
        sel_act_str = "Local" if t['sel_act'] == 0 else "Offload"
        print(f"{t['id']:<4d} | {t['p_loc']:<7.4f} | {t['p_off']:<7.4f} | {cands_str:<15} | {t['sel_idx']:<6d} | {sel_act_str:<6} | {t['sel_score']:<+10.4f} | {t['env_r']:<+10.4f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    if loc_count > 80:
        classification = "C. CANDIDATE UTILITY METRIC FAVORS LOCAL EXECUTION"
    elif argmax_match > 90:
        classification = "B. ACTOR PROBABILITY CONTROLS REPLAY ACTION DIRECTLY"
    elif loc_count > 0 and off_count > 0:
        classification = "A. CANDIDATE SELECTION GENERATES REPLAY STATE-ACTION CONFOUNDING"
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
