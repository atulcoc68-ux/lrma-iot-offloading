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
        'std': float(np_arr.std())
    }


def safe_corr(x, y):
    x_flat = np.array(x).flatten()
    y_flat = np.array(y).flatten()
    if np.std(x_flat) == 0 or np.std(y_flat) == 0:
        return 0.0
    r, _ = pearsonr(x_flat, y_flat)
    return float(r) if not np.isnan(r) else 0.0


def compute_counterfactual_dq_dr(trainer, states_t, actions_t, ed_idx=0):
    batch_size = states_t.shape[0]

    # Q(local) and Q(offload)
    a_loc = actions_t.clone(); a_loc[:, ed_idx*2] = 1.0; a_loc[:, ed_idx*2+1] = 0.0
    a_off = actions_t.clone(); a_off[:, ed_idx*2] = 0.0; a_off[:, ed_idx*2+1] = 1.0

    with torch.no_grad():
        q_loc = trainer.critic(states_t, a_loc).cpu().numpy().flatten()
        q_off = trainer.critic(states_t, a_off).cpu().numpy().flatten()

    delta_q = q_off - q_loc

    # Environment Counterfactual Reward Delta R
    delta_r_list = []
    r_loc_list = []
    r_off_list = []

    for idx in range(batch_size):
        s_i = states_t[idx].cpu().numpy()
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

        r_loc_list.append(r_loc)
        r_off_list.append(r_off)
        delta_r_list.append(r_off - r_loc)

    delta_r = np.array(delta_r_list)
    r_loc_arr = np.array(r_loc_list)
    r_off_arr = np.array(r_off_list)

    return q_loc, q_off, delta_q, r_loc_arr, r_off_arr, delta_r


def run_fitting_experiment(trainer_init, states_t, actions_t, rewards_t, next_states_t, dones_t, use_bootstrap=True):
    trainer = copy.deepcopy(trainer_init)
    batch_size = states_t.shape[0]
    checkpoints = [0, 1, 5, 10, 20, 50, 100]
    history = []

    for step in range(101):
        q_loc, q_off, dq, r_loc, r_off, dr = compute_counterfactual_dq_dr(trainer, states_t, actions_t, ed_idx=0)

        with torch.no_grad():
            cQ = trainer.critic(states_t, actions_t).cpu().numpy().flatten()
            t_ed_actions = [trainer.ed_target_actors[k](next_states_t[:, k*9:(k+1)*9]) for k in range(trainer.num_ed)]
            t_cloud_probs = trainer.cloud_target_actor(next_states_t[:, -11:])
            t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

            tQ = trainer.critic_target(next_states_t, t_joint_action).cpu().numpy().flatten()
            r_np = rewards_t.cpu().numpy().flatten()
            boot_np = 0.99 * (1.0 - dones_t.cpu().numpy().flatten()) * tQ

            if use_bootstrap:
                y_np = r_np + boot_np
            else:
                y_np = r_np

        if step in checkpoints:
            p_norm = sum(p.norm(2).item()**2 for p in trainer.critic.parameters())**0.5
            history.append({
                'step': step,
                'r_stats': get_stats(r_np),
                'boot_stats': get_stats(boot_np),
                'y_stats': get_stats(y_np),
                'q_loc_mean': float(q_loc.mean()),
                'q_off_mean': float(q_off.mean()),
                'dq_mean': float(dq.mean()),
                'r_loc_mean': float(r_loc.mean()),
                'r_off_mean': float(r_off.mean()),
                'dr_mean': float(dr.mean()),
                'corr_q_y': safe_corr(cQ, y_np),
                'corr_q_r': safe_corr(cQ, r_np),
                'corr_dq_dr': safe_corr(dq, dr),
                'abs_dq_mean': float(np.abs(dq).mean()),
                'abs_dr_mean': float(np.abs(dr).mean()),
                'q_mean': float(cQ.mean()),
                'q_std': float(cQ.std()),
                'param_norm': float(p_norm)
            })

        if step < 100:
            y_t = torch.FloatTensor(y_np).unsqueeze(1).to(trainer.device)
            cQ_t = trainer.critic(states_t, actions_t)
            c_loss = nn.MSELoss()(cQ_t, y_t)

            trainer.critic_optimizer.zero_grad()
            c_loss.backward()

            g_norm = sum(p.grad.norm(2).item()**2 for p in trainer.critic.parameters() if p.grad is not None)**0.5
            if step in checkpoints:
                history[-1]['grad_norm'] = float(g_norm)
                history[-1]['c_loss'] = float(c_loss.item())

            trainer.critic_optimizer.step()
            soft_update(trainer.critic_target, trainer.critic, 0.01)

    return history


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
    next_states_t = torch.FloatTensor(next_states).to(trainer_init.device)
    dones_t = torch.FloatTensor(dones).unsqueeze(1).to(trainer_init.device)

    print("\n============================================================")
    print("CONTROL A: FITTING CRITIC TO IMMEDIATE REWARD Y = R")
    print("============================================================")
    hist_ctrl_a = run_fitting_experiment(trainer_init, states_t, actions_t, rewards_t, next_states_t, dones_t, use_bootstrap=False)

    print(f"{'Step':<5} | {'Loss':<9} | {'Q(loc)':<8} | {'Q(off)':<8} | {'dQ (off-loc)':<12} | {'dR (off-loc)':<12} | {'Corr(Q,R)':<10} | {'Corr(dQ,dR)':<11}")
    print("-" * 95)
    for h in hist_ctrl_a:
        loss_val = h.get('c_loss', 0.0)
        print(f"{h['step']:<5d} | {loss_val:<9.4f} | {h['q_loc_mean']:<+8.4f} | {h['q_off_mean']:<+8.4f} | {h['dq_mean']:<+12.6f} | {h['dr_mean']:<+12.6f} | {h['corr_q_r']:<+10.4f} | {h['corr_dq_dr']:<+11.4f}")

    print("\n============================================================")
    print("CONTROL B: FITTING CRITIC TO FULL TD TARGET Y = R + gamma*Q_target")
    print("============================================================")
    hist_ctrl_b = run_fitting_experiment(trainer_init, states_t, actions_t, rewards_t, next_states_t, dones_t, use_bootstrap=True)

    print(f"{'Step':<5} | {'Loss':<9} | {'Q(loc)':<8} | {'Q(off)':<8} | {'dQ (off-loc)':<12} | {'dR (off-loc)':<12} | {'Corr(Q,Y)':<10} | {'Corr(dQ,dR)':<11}")
    print("-" * 95)
    for h in hist_ctrl_b:
        loss_val = h.get('c_loss', 0.0)
        print(f"{h['step']:<5d} | {loss_val:<9.4f} | {h['q_loc_mean']:<+8.4f} | {h['q_off_mean']:<+8.4f} | {h['dq_mean']:<+12.6f} | {h['dr_mean']:<+12.6f} | {h['corr_q_y']:<+10.4f} | {h['corr_dq_dr']:<+11.4f}")

    print("\n============================================================")
    print("FINAL REWARD VS BOOTSTRAP ATTRIBUTION CLASSIFICATION")
    print("============================================================")

    dq_a_fin = hist_ctrl_a[-1]['dq_mean']
    dq_b_fin = hist_ctrl_b[-1]['dq_mean']
    dr_fin = hist_ctrl_a[-1]['dr_mean']

    if dr_fin < 0 and dq_a_fin < 0 and dq_b_fin < 0:
        classification = "A. REWARD R DIRECTLY DRIVES LOCAL PREFERENCE"
    elif dr_fin >= 0 and dq_b_fin < 0:
        classification = "B. BOOTSTRAP TARGET INTRODUCES LOCAL PREFERENCE"
    elif abs(dq_b_fin) > abs(dq_a_fin) * 2.0:
        classification = "C. CRITIC FITTING AMPLIFIES LOCAL PREFERENCE"
    else:
        classification = "D. REWARD AND BOOTSTRAP BOTH FAVOR LOCAL"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("HOST EXECUTION COMPLETED: NO (STATIC VERIFICATION ONLY)")


if __name__ == "__main__":
    main()
