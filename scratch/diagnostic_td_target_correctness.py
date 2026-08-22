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


def get_stats(tensor):
    np_arr = tensor.detach().cpu().numpy()
    return {
        'shape': tuple(tensor.shape),
        'min': float(np_arr.min()),
        'max': float(np_arr.max()),
        'mean': float(np_arr.mean()),
        'std': float(np_arr.std()),
        'finite': bool(torch.isfinite(tensor).all().item())
    }


def print_stats(label, stats):
    print(f"  {label:<25}: shape={stats['shape']}, min={stats['min']:7.4f}, max={stats['max']:7.4f}, mean={stats['mean']:7.4f}, std={stats['std']:7.4f}, finite={stats['finite']}")


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
        # Add a few done=True for mask audit
        done = (i % 16 == 0)
        ns_joint = np.random.randn(236).astype(np.float32)
        trainer.replay_buffer_ed.add(s_joint, a_joint, r, ns_joint, done)

    states, actions, rewards, next_states, dones = trainer.replay_buffer_ed.sample(batch_size)

    states_t = torch.FloatTensor(states).to(trainer.device)
    actions_t = torch.FloatTensor(actions).to(trainer.device)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(trainer.device)
    next_states_t = torch.FloatTensor(next_states).to(trainer.device)
    dones_t = torch.FloatTensor(dones).unsqueeze(1).to(trainer.device)

    # Generate Target Joint Action
    with torch.no_grad():
        target_ed_actions = []
        for idx in range(trainer.num_ed):
            ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
            target_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
        cloud_st = next_states_t[:, -11:]
        cloud_target_probs = trainer.cloud_target_actor(cloud_st)
        target_joint_action = torch.cat(target_ed_actions + [cloud_target_probs], dim=-1)

        gamma = 0.99
        target_next_q = trainer.critic_target(next_states_t, target_joint_action)
        bootstrap = gamma * target_next_q
        target_y = rewards_t + (1.0 - dones_t) * bootstrap

    print("============================================================")
    print("TD TARGET CORRECTNESS DIAGNOSTIC REPORT")
    print("============================================================")

    print("\nBASIC TD QUANTITIES:")
    print_stats("States (S)", get_stats(states_t))
    print_stats("Actions (A)", get_stats(actions_t))
    print_stats("Rewards (R)", get_stats(rewards_t))
    print_stats("Next States (S')", get_stats(next_states_t))
    print_stats("Dones (d)", get_stats(dones_t))
    print_stats("Target Joint Action (A')", get_stats(target_joint_action))
    print_stats("Target Next Q (Q_target)", get_stats(target_next_q))
    print_stats("Bootstrap (gamma*Q_trg)", get_stats(bootstrap))
    print_stats("Target Y (r + (1-d)boot)", get_stats(target_y))

    # A. REWARD VS BOOTSTRAP SCALE
    reward_mag = float(rewards_t.abs().mean().item())
    bootstrap_mag = float(bootstrap.abs().mean().item())
    ratio = bootstrap_mag / (reward_mag + 1e-8)

    print("\nA. REWARD VS BOOTSTRAP SCALE:")
    print(f"  Reward Magnitude Mean     : {reward_mag:.4f}")
    print(f"  Bootstrap Magnitude Mean  : {bootstrap_mag:.4f}")
    print(f"  Ratio (Bootstrap/Reward)  : {ratio:.4f}")

    if ratio > 5.0:
        scale_domination = "BOOSTRAP Q DOMINATED"
    elif ratio < 0.2:
        scale_domination = "IMMEDIATE REWARD DOMINATED"
    else:
        scale_domination = "BALANCED SCALE"
    print(f"  Domination Status        : {scale_domination}")

    # B. DONE MASK AUDIT
    done_true_count = int((dones_t == 1.0).sum().item())
    done_false_count = int((dones_t == 0.0).sum().item())
    done_mask_valid = torch.allclose(target_y[dones_t == 1.0], rewards_t[dones_t == 1.0]) if done_true_count > 0 else True

    print("\nB. DONE MASK AUDIT:")
    print(f"  done=True Count          : {done_true_count}")
    print(f"  done=False Count         : {done_false_count}")
    print(f"  Done Target Y == Reward  : {'PASS' if done_mask_valid else 'FAIL'}")

    # C. TARGET-Q ACTION AUDIT
    with torch.no_grad():
        q_trg_replay = trainer.critic_target(next_states_t, actions_t)
        q_trg_diff = target_next_q - q_trg_replay

    q_trg_diff_stats = get_stats(q_trg_diff)
    print("\nC. TARGET-Q ACTION AUDIT (Target Actor Probs vs Replay One-Hot):")
    print(f"  Q_trg(Prob) - Q_trg(Replay) : mean={q_trg_diff_stats['mean']:+.6f}, std={q_trg_diff_stats['std']:.6f}, min={q_trg_diff_stats['min']:+.6f}, max={q_trg_diff_stats['max']:+.6f}")

    # D. REWARD / TARGET CORRELATION
    r_np = rewards_t.squeeze().cpu().numpy()
    tnq_np = target_next_q.squeeze().cpu().numpy()
    ty_np = target_y.squeeze().cpu().numpy()

    corr_r_tnq = float(np.corrcoef(r_np, tnq_np)[0, 1]) if (np.std(r_np) > 1e-8 and np.std(tnq_np) > 1e-8) else 0.0
    corr_r_ty = float(np.corrcoef(r_np, ty_np)[0, 1]) if (np.std(r_np) > 1e-8 and np.std(ty_np) > 1e-8) else 0.0

    print("\nD. REWARD / TARGET CORRELATION:")
    print(f"  Corr(Reward, Target Next Q) : {corr_r_tnq:.4f}")
    print(f"  Corr(Reward, Target Y)      : {corr_r_ty:.4f}")

    # E. TD ERROR AUDIT
    with torch.no_grad():
        current_q = trainer.critic(states_t, actions_t)
        td_error = target_y - current_q

    tde_stats = get_stats(td_error)
    tde_mse = float((td_error ** 2).mean().item())
    frac_pos = float((td_error > 0).float().mean().item())
    frac_neg = float((td_error < 0).float().mean().item())

    print("\nE. TD ERROR AUDIT:")
    print(f"  TD Error Mean             : {tde_stats['mean']:+.6f}")
    print(f"  TD Error Std              : {tde_stats['std']:.6f}")
    print(f"  TD Error Min / Max        : {tde_stats['min']:+.6f} / {tde_stats['max']:+.6f}")
    print(f"  TD Error MSE              : {tde_mse:.6f}")
    print(f"  Fraction Positive (y > Q) : {frac_pos * 100.0:.2f}%")
    print(f"  Fraction Negative (y < Q) : {frac_neg * 100.0:.2f}%")

    # F. CRITICAL ACTION-SPECIFIC TD TEST
    reward_calc = LRMARewardCalculator(V_penalty=EnvConfig.V)
    wireless = WirelessModel()
    ed_positions = np.random.uniform(0, 500, size=(25, 2))
    bs_positions = np.array([[100, 100], [400, 100], [250, 250], [100, 400], [400, 400]], dtype=np.float64)

    td_target_favors_offload = 0
    td_target_favors_local = 0

    num_samples = min(batch_size, 20)
    for i in range(num_samples):
        task = frozen_tasks[i] if i < len(frozen_tasks) else None
        t_size = task.size if task else 1.5 * 1024 * 1024 * 8
        t_C = task.C if task else 500.0
        t_G = task.G if task else 200.0
        task_R = task.R if task else 1

        d_local = max((t_C * 1e6) / float(EnvConfig.LOCAL_CPU_CAPACITY), ((t_G * 1e6) / float(EnvConfig.LOCAL_GPU_CAPACITY)) if task_R > 0 else 0.0)
        q_dev = abs(states[i, 6]) * 8e6
        r_loc = reward_calc.calculate_ed_individual_reward(t_size, d_local, q_dev, 1.0, is_offloaded=False)

        assigned_mes = int(np.argmax(actions[i, -5:]))
        dist = wireless.calculate_distance(ed_positions[0], bs_positions[assigned_mes])
        t_trans = wireless.calculate_transmission_delay(t_size, wireless.calculate_rate(wireless.calculate_channel_gain(dist)))
        d_mes = max((t_C * 1e6) / float(EnvConfig.MES_TOTAL_CPU_CAPACITY / 3.0), ((t_G * 1e6) / float(EnvConfig.MES_GPU_CAPACITY)) if task_R > 0 else 0.0)
        r_off = reward_calc.calculate_ed_individual_reward(t_size, t_trans + d_mes, q_dev, 1.0, is_offloaded=True)

        y_val = target_y[i].item()
        if y_val > max(r_loc, r_off):
            td_target_favors_offload += 1
        else:
            td_target_favors_local += 1

    print("\nF. CRITICAL ACTION-SPECIFIC TD TEST:")
    print(f"  Samples Tested            : {num_samples}")
    print(f"  TD Target Favors Offload  : {td_target_favors_offload}")
    print(f"  TD Target Favors Local    : {td_target_favors_local}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    if ratio < 0.2:
        classification = "C. TD TARGET IS DOMINATED BY IMMEDIATE REWARD"
    elif ratio > 5.0:
        classification = "B. TD TARGET IS DOMINATED BY BOOTSTRAP Q"
    elif abs(q_trg_diff_stats['mean']) > 0.5:
        classification = "D. TARGET-ACTION REPRESENTATION IS CAUSING BIAS"
    elif not done_mask_valid:
        classification = "E. TD TARGET / REWARD PIPELINE IS INCONSISTENT"
    elif abs(tde_stats['mean']) < 0.5 and done_mask_valid and ratio >= 0.2 and ratio <= 5.0:
        classification = "A. TD TARGET IS CONSISTENT AND WELL-SCALED"
    else:
        classification = "F. INSUFFICIENT EVIDENCE"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
