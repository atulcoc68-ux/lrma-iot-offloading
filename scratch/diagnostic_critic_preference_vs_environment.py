import sys
import os
import json
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from src.lrma_trainer import LRMATrainer
from src.data_loader import LRMATask
from src.lyapunov import LRMARewardCalculator
from src.wireless import WirelessModel
from src.config import EnvConfig


def load_frozen_trace_tasks(trace_path):
    if not os.path.exists(trace_path):
        return []
    with open(trace_path, 'r') as f:
        data = json.load(f)
    task_list = data.get('tasks', [])
    return [LRMATask.from_dict(t) for t in task_list]


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)

    trace_path = "results/raw/workload_trace_train_seed42_N25_rate60.json"
    frozen_tasks = load_frozen_trace_tasks(trace_path)
    if frozen_tasks:
        print(f"Loaded {len(frozen_tasks)} frozen tasks from {trace_path}")

    # Build 64 replay samples
    for i in range(batch_size):
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

    # Initialize Environment & Reward Calculator for isolated step evaluations
    reward_calc = LRMARewardCalculator(V_penalty=EnvConfig.V)
    wireless = WirelessModel()

    ed_positions = np.random.uniform(0, 500, size=(25, 2))
    bs_positions = np.array([
        [100, 100], [400, 100], [250, 250], [100, 400], [400, 400]
    ], dtype=np.float64)

    critic_preferences = []
    env_preferences = []
    agree_count = 0
    disagree_count = 0
    failure_case_count = 0  # Critic prefers OFFLOAD (dQ > 0) but Environment prefers LOCAL (dR < 0)

    num_samples = min(batch_size, 50)

    print("\n============================================================")
    print("CRITIC PREFERENCE VS ENVIRONMENT REWARD DIAGNOSTIC")
    print("============================================================")

    for i in range(num_samples):
        # 1. Alternative A: ED-0 Local [1, 0]
        a_local = actions_t[i:i+1].clone()
        a_local[0, 0] = 1.0
        a_local[0, 1] = 0.0

        # 2. Alternative B: ED-0 Offload [0, 1]
        a_offload = actions_t[i:i+1].clone()
        a_offload[0, 0] = 0.0
        a_offload[0, 1] = 1.0

        with torch.no_grad():
            q_local = trainer.critic(states_t[i:i+1], a_local).item()
            q_offload = trainer.critic(states_t[i:i+1], a_offload).item()

        dq = q_offload - q_local
        critic_preferences.append(dq)

        # Environment Evaluation (Isolated task environment computation)
        task = frozen_tasks[i] if i < len(frozen_tasks) else None
        t_size = task.size if task else 1.5 * 1024 * 1024 * 8  # bits
        t_C = task.C if task else 500.0  # M cycles
        t_G = task.G if task else 200.0
        task_R = task.R if task else 1

        # Local execution delay & reward
        c_time_loc = (t_C * 1e6) / float(EnvConfig.LOCAL_CPU_CAPACITY)
        g_time_loc = ((t_G * 1e6) / float(EnvConfig.LOCAL_GPU_CAPACITY)) if task_R > 0 else 0.0
        d_local = max(c_time_loc, g_time_loc)

        q_dev = abs(states[i, 6]) * 8e6
        q_tilde = 1.0
        r_local = reward_calc.calculate_ed_individual_reward(t_size, d_local, q_dev, q_tilde, is_offloaded=False)

        # Offload execution delay & reward
        assigned_mes = int(np.argmax(actions[i, -5:]))
        dist = wireless.calculate_distance(ed_positions[0], bs_positions[assigned_mes])
        h_gain = wireless.calculate_channel_gain(dist)
        v_rate = wireless.calculate_rate(h_gain)
        t_trans = wireless.calculate_transmission_delay(t_size, v_rate)

        c_time_mes = (t_C * 1e6) / float(EnvConfig.MES_TOTAL_CPU_CAPACITY / 3.0)
        g_time_mes = ((t_G * 1e6) / float(EnvConfig.MES_GPU_CAPACITY)) if task_R > 0 else 0.0
        d_mes = max(c_time_mes, g_time_mes)
        d_offload = t_trans + d_mes

        r_offload = reward_calc.calculate_ed_individual_reward(t_size, d_offload, q_dev, q_tilde, is_offloaded=True)

        dr = r_offload - r_local
        env_preferences.append(dr)

        # Agreement / Disagreement analysis
        agree = (dq * dr > 0) or (abs(dq) < 1e-6 and abs(dr) < 1e-6)
        if agree:
            agree_count += 1
        else:
            disagree_count += 1

        # Failure case: Critic prefers OFFLOAD (dq > 0), but Env prefers LOCAL (dr < 0)
        if dq > 1e-4 and dr < -1e-4:
            failure_case_count += 1

    critic_prefs_np = np.array(critic_preferences)
    env_prefs_np = np.array(env_preferences)

    mean_c_pref = float(np.mean(critic_prefs_np))
    mean_e_pref = float(np.mean(env_prefs_np))
    median_c_pref = float(np.median(critic_prefs_np))
    median_e_pref = float(np.median(env_prefs_np))

    agreement_pct = (agree_count / num_samples) * 100.0

    if np.std(critic_prefs_np) > 1e-8 and np.std(env_prefs_np) > 1e-8:
        corr = float(np.corrcoef(critic_prefs_np, env_prefs_np)[0, 1])
    else:
        corr = 0.0

    print(f"Evaluated Samples                : {num_samples}")
    print(f"Mean Critic Preference (dQ)      : {mean_c_pref:.6f}")
    print(f"Mean Environment Preference (dR) : {mean_e_pref:.6f}")
    print(f"Median Critic Preference (dQ)    : {median_c_pref:.6f}")
    print(f"Median Environment Preference (dR): {median_e_pref:.6f}")
    print(f"Cases Agreed                     : {agree_count}")
    print(f"Cases Disagreed                  : {disagree_count}")
    print(f"Agreement Percentage             : {agreement_pct:.2f}%")
    print(f"Preference Correlation           : {corr:.4f}")
    print(f"Failure Cases (Critic Offload, Env Local): {failure_case_count}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    if agreement_pct >= 70.0:
        classification = "A. CRITIC PREFERENCE AGREES WITH ENVIRONMENT"
    elif failure_case_count > num_samples * 0.4:
        classification = "B. CRITIC PREFERENCE CONFLICTS WITH ENVIRONMENT"
    elif 40.0 <= agreement_pct < 70.0:
        classification = "C. MIXED / STATE-DEPENDENT PREFERENCE"
    else:
        classification = "D. INSUFFICIENT EVIDENCE"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
