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


def compute_delta_q(critic_net, states_t, actions_t, ed_idx=0):
    a_loc = actions_t.clone()
    a_loc[:, ed_idx*2] = 1.0
    a_loc[:, ed_idx*2+1] = 0.0

    a_off = actions_t.clone()
    a_off[:, ed_idx*2] = 0.0
    a_off[:, ed_idx*2+1] = 1.0

    with torch.no_grad():
        q_loc = critic_net(states_t, a_loc).cpu().numpy().flatten()
        q_off = critic_net(states_t, a_off).cpu().numpy().flatten()

    return q_off - q_loc


def eval_landscape(critic_net, states_t, actions_t, ed_idx=0):
    alphas = [0.0, 0.25, 0.50, 0.75, 1.0]
    q_means = []
    for alpha in alphas:
        a_interp = actions_t.clone()
        a_interp[:, ed_idx*2] = 1.0 - alpha
        a_interp[:, ed_idx*2+1] = alpha
        with torch.no_grad():
            q_m = critic_net(states_t, a_interp).mean().item()
            q_means.append(q_m)
    slope = q_means[-1] - q_means[0]
    return q_means, slope


def run_trajectory_experiment(trainer_init, states_t, actions_t, rewards_t, next_states_t, dones_t, freeze_actors=False):
    trainer = copy.deepcopy(trainer_init)
    batch_size = states_t.shape[0]

    history = []

    delta_r_list = []
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

        delta_r_list.append(r_off - r_loc)

    delta_r = np.array(delta_r_list)

    def record_checkpoint(step_num):
        dq_prim_ed0 = compute_delta_q(trainer.critic, states_t, actions_t, ed_idx=0)
        dq_trg_ed0 = compute_delta_q(trainer.critic_target, states_t, actions_t, ed_idx=0)

        dq_all_eds = []
        for e in range(trainer.num_ed):
            dq_e = compute_delta_q(trainer.critic, states_t, actions_t, ed_idx=e)
            dq_all_eds.append(dq_e.mean())

        mean_dq_eds = float(np.mean(dq_all_eds))
        pct_dq_pos = float((dq_prim_ed0 > 0).sum() / batch_size) * 100.0

        q_landscape, slope = eval_landscape(trainer.critic, states_t, actions_t, ed_idx=0)

        with torch.no_grad():
            cQ = trainer.critic(states_t, actions_t).cpu().numpy().flatten()

            t_ed_actions = []
            for idx in range(trainer.num_ed):
                ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                t_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
            cloud_st = next_states_t[:, -11:]
            t_cloud_probs = trainer.cloud_target_actor(cloud_st)
            t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

            tQ = trainer.critic_target(next_states_t, t_joint_action).cpu().numpy().flatten()
            ty = (rewards_t + (1.0 - dones_t) * 0.99 * torch.FloatTensor(tQ).unsqueeze(1).to(trainer.device)).cpu().numpy().flatten()

            td_err = cQ - ty

        sign_agree = np.sign(dq_prim_ed0) == np.sign(delta_r)
        agree_pct = float(sign_agree.sum() / batch_size) * 100.0
        off_loc_fail = int(((dq_prim_ed0 > 0) & (delta_r < 0)).sum())

        return {
            'step': step_num,
            'cQ_stats': get_stats(cQ),
            'tQ_stats': get_stats(tQ),
            'ty_stats': get_stats(ty),
            'td_err_stats': get_stats(td_err),
            'dq_prim_ed0_stats': get_stats(dq_prim_ed0),
            'dq_trg_ed0_stats': get_stats(dq_trg_ed0),
            'mean_dq_eds': mean_dq_eds,
            'pct_dq_pos': pct_dq_pos,
            'agree_pct': agree_pct,
            'off_loc_fail': off_loc_fail,
            'q_landscape': q_landscape,
            'slope': slope,
            'dq_prim_raw': dq_prim_ed0,
            'dq_trg_raw': dq_trg_ed0,
            'delta_r_raw': delta_r,
            'tQ_raw': tQ,
            'ty_raw': ty
        }

    history.append(record_checkpoint(0))

    for step in range(1, 101):
        if freeze_actors:
            with torch.no_grad():
                t_ed_actions = []
                for idx in range(trainer.num_ed):
                    ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                    t_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
                cloud_st = next_states_t[:, -11:]
                t_cloud_probs = trainer.cloud_target_actor(cloud_st)
                t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)

                tQ = trainer.critic_target(next_states_t, t_joint_action)
                ty = rewards_t + (1.0 - dones_t) * 0.99 * tQ

            cQ = trainer.critic(states_t, actions_t)
            c_loss = nn.MSELoss()(cQ, ty)

            trainer.critic_optimizer.zero_grad()
            c_loss.backward()
            trainer.critic_optimizer.step()

            soft_update(trainer.critic_target, trainer.critic, 0.01)
        else:
            trainer.train_step(batch_size=batch_size, gamma=0.99, xi_soft=0.01)

        if step % 5 == 0:
            history.append(record_checkpoint(step))

    return history, delta_r


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

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
    print("PART 1 & PART 2: INITIAL BASELINE & ENVIRONMENT REFERENCE")
    print("============================================================")

    dq_init = compute_delta_q(trainer_init.critic, states_t, actions_t, ed_idx=0)
    dq_init_st = get_stats(dq_init)

    print(f"Initial Critic ED-0 DeltaQ (Offload - Local):")
    print(f"  Mean: {dq_init_st['mean']:+.6f}, Median: {dq_init_st['median']:+.6f}, Std: {dq_init_st['std']:.6f}, Min: {dq_init_st['min']:+.6f}, Max: {dq_init_st['max']:+.6f}")
    print(f"  Percentage DeltaQ > 0: {(dq_init > 0).sum() / batch_size * 100.0:.1f}%")

    print("\n============================================================")
    print("PART 6: ACTOR-FROZEN CONTROL (100 CRITIC UPDATES)")
    print("============================================================")
    hist_critic_only, delta_r = run_trajectory_experiment(trainer_init, states_t, actions_t, rewards_t, next_states_t, dones_t, freeze_actors=True)

    print(f"{'Step':<5} | {'Mean Q':<9} | {'Q Std':<8} | {'ED0 DeltaQ':<11} | {'% DeltaQ>0':<11} | {'Trg DeltaQ':<11} | {'Agree %':<8} | {'Off-Fail':<8}")
    print("-" * 88)

    for h in hist_critic_only:
        if h['step'] % 20 == 0 or h['step'] in [0, 5, 10]:
            print(f"{h['step']:<5d} | {h['cQ_stats']['mean']:<+9.4f} | {h['cQ_stats']['std']:<8.4f} | {h['dq_prim_ed0_stats']['mean']:<+11.6f} | {h['pct_dq_pos']:<11.1f} | {h['dq_trg_ed0_stats']['mean']:<+11.6f} | {h['agree_pct']:<8.1f} | {h['off_loc_fail']:<8d}")

    print("\n============================================================")
    print("PART 7: FULL CTDE CONTROL (100 CTDE UPDATES)")
    print("============================================================")
    hist_full_ctde, _ = run_trajectory_experiment(trainer_init, states_t, actions_t, rewards_t, next_states_t, dones_t, freeze_actors=False)

    print(f"{'Step':<5} | {'Mean Q':<9} | {'Q Std':<8} | {'ED0 DeltaQ':<11} | {'% DeltaQ>0':<11} | {'Trg DeltaQ':<11} | {'Agree %':<8} | {'Off-Fail':<8}")
    print("-" * 88)

    for h in hist_full_ctde:
        if h['step'] % 20 == 0 or h['step'] in [0, 5, 10]:
            print(f"{h['step']:<5d} | {h['cQ_stats']['mean']:<+9.4f} | {h['cQ_stats']['std']:<8.4f} | {h['dq_prim_ed0_stats']['mean']:<+11.6f} | {h['pct_dq_pos']:<11.1f} | {h['dq_trg_ed0_stats']['mean']:<+11.6f} | {h['agree_pct']:<8.1f} | {h['off_loc_fail']:<8d}")

    print("\n============================================================")
    print("PART 8: CRITIC ACTION LANDSCAPE EVOLUTION (ED-0)")
    print("============================================================")
    print(f"{'Step':<5} | {'Q(a=0.0)':<10} | {'Q(a=0.25)':<10} | {'Q(a=0.50)':<10} | {'Q(a=0.75)':<10} | {'Q(a=1.0)':<10} | {'FD Slope':<10}")
    print("-" * 80)
    for h in hist_full_ctde:
        if h['step'] in [0, 5, 10, 20, 50, 100]:
            ql = h['q_landscape']
            print(f"{h['step']:<5d} | {ql[0]:<+10.4f} | {ql[1]:<+10.4f} | {ql[2]:<+10.4f} | {ql[3]:<+10.4f} | {ql[4]:<+10.4f} | {h['slope']:<+10.6f}")

    print("\n============================================================")
    print("PART 9: TD TARGET DECOMPOSITION TRAJECTORY")
    print("============================================================")
    print(f"{'Step':<5} | {'Mean |R|':<10} | {'Mean |gamma*tQ|':<15} | {'Boot/Reward Ratio':<18} | {'Corr(R, tQ)':<12} | {'Corr(R, Y)':<12}")
    print("-" * 80)

    for h in hist_full_ctde:
        if h['step'] % 20 == 0 or h['step'] in [0, 5, 10]:
            r_abs_m = np.abs(rewards).mean()
            tq_abs_m = np.abs(0.99 * h['tQ_raw']).mean()
            ratio = tq_abs_m / r_abs_m if r_abs_m > 0 else 0.0
            c_r_tq, _ = pearsonr(rewards, h['tQ_raw'])
            c_r_y, _ = pearsonr(rewards, h['ty_raw'])
            print(f"{h['step']:<5d} | {r_abs_m:<10.4f} | {tq_abs_m:<15.4f} | {ratio:<18.4f} | {c_r_tq:<+12.4f} | {c_r_y:<+12.4f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    h_fin_critic = hist_critic_only[-1]
    h_fin_full = hist_full_ctde[-1]

    no_offload_bias = (h_fin_critic['pct_dq_pos'] < 50.0 and h_fin_full['pct_dq_pos'] < 50.0)

    if no_offload_bias:
        classification = "G. NO CRITIC PREFERENCE TRANSITION OBSERVED"
    elif h_fin_critic['pct_dq_pos'] > 50.0 and h_fin_full['pct_dq_pos'] > 50.0:
        classification = "B. CRITIC DEVELOPS OFFLOAD BIAS DURING TRAINING"
    elif abs(h_fin_critic['dq_prim_ed0_stats']['mean'] - h_fin_critic['dq_trg_ed0_stats']['mean']) > 0.5:
        classification = "E. TARGET CRITIC LAG CREATES OFFLOAD BIAS"
    elif h_fin_full['pct_dq_pos'] > 50.0 and h_fin_critic['pct_dq_pos'] < 50.0:
        classification = "F. ACTOR MOVEMENT CAUSES CRITIC BIAS"
    else:
        classification = "H. INCONCLUSIVE"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("REAL TRAINER MODIFIED: NO")


if __name__ == "__main__":
    main()
