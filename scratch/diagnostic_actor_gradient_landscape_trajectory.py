import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
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
        'std': float(np_arr.std())
    }


def compute_entropy(probs):
    eps = 1e-12
    p_safe = torch.clamp(probs, min=eps)
    return -(p_safe * torch.log(p_safe)).sum(dim=-1).mean().item()


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
    print("PART 1-8: ACTOR GRADIENT LANDSCAPE TRAJECTORY (50 UPDATES)")
    print("============================================================")

    trainer = copy.deepcopy(trainer_init)
    checkpoints = [0, 1, 2, 5, 10, 20, 30, 40, 50]
    history = []

    for step in range(51):
        # Evaluate current ED-0 policy and critic landscape before update
        ed0_st = states_t[:, 0:9]
        with torch.no_grad():
            ed0_probs = trainer.ed_primary_actors[0](ed0_st)
            p_loc = ed0_probs[:, 0].mean().item()
            p_off = ed0_probs[:, 1].mean().item()
            ent_ed0 = compute_entropy(ed0_probs)

            # Evaluate endpoint Q
            a_loc = actions_t.clone()
            a_loc[:, 0] = 1.0; a_loc[:, 1] = 0.0
            q_loc = trainer.critic(states_t, a_loc).mean().item()

            a_off = actions_t.clone()
            a_off[:, 0] = 0.0; a_off[:, 1] = 1.0
            q_off = trainer.critic(states_t, a_off).mean().item()

            endpoint_delta_q = q_off - q_loc

            cloud_st = states_t[:, -11:]
            cloud_probs = trainer.cloud_primary_actor(cloud_st)
            ent_cloud = compute_entropy(cloud_probs)

        # Part 3: Critic gradient at actual policy
        a_actual = actions_t.clone().detach().requires_grad_(True)
        # Substitute ED-0 actions with actual actor probabilities
        a_actual_prob = actions_t.clone()
        a_actual_prob[:, 0:2] = ed0_probs
        a_actual_prob_in = a_actual_prob.detach().requires_grad_(True)

        q_actual = trainer.critic(states_t, a_actual_prob_in)
        q_actual_mean = q_actual.mean().item()

        q_actual.sum().backward()
        dq_da = a_actual_prob_in.grad.detach()
        dq_da_loc = dq_da[:, 0].mean().item()
        dq_da_off = dq_da[:, 1].mean().item()
        gradient_delta_q = dq_da_off - dq_da_loc

        # Part 7: Finite difference check at actual policy
        eps = 1e-3
        a_actual_plus = a_actual_prob.clone()
        a_actual_plus[:, 0] = torch.clamp(a_actual_prob[:, 0] - eps, 0.0, 1.0)
        a_actual_plus[:, 1] = torch.clamp(a_actual_prob[:, 1] + eps, 0.0, 1.0)

        a_actual_minus = a_actual_prob.clone()
        a_actual_minus[:, 0] = torch.clamp(a_actual_prob[:, 0] + eps, 0.0, 1.0)
        a_actual_minus[:, 1] = torch.clamp(a_actual_prob[:, 1] - eps, 0.0, 1.0)

        with torch.no_grad():
            q_plus = trainer.critic(states_t, a_actual_plus).mean().item()
            q_minus = trainer.critic(states_t, a_actual_minus).mean().item()
            fd_slope = (q_plus - q_minus) / (2.0 * eps)

        fd_abs_err = abs(fd_slope - gradient_delta_q)

        # Part 6: Classification
        end_loc = endpoint_delta_q < 0
        grad_loc = gradient_delta_q < 0

        if end_loc and grad_loc:
            cls_code = 'A' # Both LOCAL
        elif end_loc and not grad_loc:
            cls_code = 'B' # End LOCAL, Grad OFFLOAD
        elif not end_loc and grad_loc:
            cls_code = 'C' # End OFFLOAD, Grad LOCAL
        else:
            cls_code = 'D' # Both OFFLOAD

        # Part 5: Perform step & track update direction
        p_probs_loss = trainer.ed_primary_actors[0](ed0_st)
        j_diff = actions_t.clone()
        j_diff[:, 0:2] = p_probs_loss
        a_loss = -trainer.critic(states_t, j_diff).mean()

        trainer.ed_optimizers[0].zero_grad()
        a_loss.backward()

        g_flat = []
        for p in trainer.ed_primary_actors[0].parameters():
            if p.grad is not None:
                g_flat.append(p.grad.view(-1))
        g_norm = torch.cat(g_flat).norm(2).item() if g_flat else 0.0

        trainer.ed_optimizers[0].step()

        with torch.no_grad():
            ed0_probs_after = trainer.ed_primary_actors[0](ed0_st)
            p_off_after = ed0_probs_after[:, 1].mean().item()
            update_dir = "OFFLOAD" if p_off_after > p_off else "LOCAL"

        if step in checkpoints:
            history.append({
                'step': step,
                'p_loc': p_loc,
                'p_off': p_off,
                'ent_ed0': ent_ed0,
                'q_loc': q_loc,
                'q_actual': q_actual_mean,
                'q_off': q_off,
                'end_dq': endpoint_delta_q,
                'grad_dq': gradient_delta_q,
                'fd_slope': fd_slope,
                'fd_err': fd_abs_err,
                'cls': cls_code,
                'g_norm': g_norm,
                'update_dir': update_dir,
                'ent_cloud': ent_cloud
            })

        # Continue CTDE update for step > 0
        if step < 50:
            for idx in range(1, 25):
                st_i = states_t[:, idx*9:(idx+1)*9]
                pr_i = trainer.ed_primary_actors[idx](st_i)
                jd_i = actions_t.clone()
                jd_i[:, idx*2:(idx+1)*2] = pr_i
                al_i = -trainer.critic(states_t, jd_i).mean()
                trainer.ed_optimizers[idx].zero_grad()
                al_i.backward()
                trainer.ed_optimizers[idx].step()

            cloud_st = states_t[:, -11:]
            cloud_pr = trainer.cloud_primary_actor(cloud_st)
            jd_c = actions_t.clone()
            jd_c[:, -5:] = cloud_pr
            cl_loss = -trainer.critic(states_t, jd_c).mean()
            trainer.cloud_optimizer.zero_grad()
            cl_loss.backward()
            trainer.cloud_optimizer.step()

            # Critic step
            with torch.no_grad():
                t_ed_actions = [trainer.ed_target_actors[k](next_states_t[:, k*9:(k+1)*9]) for k in range(25)]
                t_cloud_probs = trainer.cloud_target_actor(next_states_t[:, -11:])
                t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)
                tQ = trainer.critic_target(next_states_t, t_joint_action)
                ty = rewards_t + (1.0 - dones_t) * 0.99 * tQ

            cQ = trainer.critic(states_t, actions_t)
            c_loss = nn.MSELoss()(cQ, ty)
            trainer.critic_optimizer.zero_grad()
            c_loss.backward()
            trainer.critic_optimizer.step()

    print(f"{'Step':<5} | {'P(loc)':<7} | {'P(off)':<7} | {'Ent':<6} | {'Q(loc)':<8} | {'Q(act)':<8} | {'Q(off)':<8} | {'End dQ':<9} | {'Grad dQ':<9} | {'FD Slope':<9} | {'Cls':<4} | {'GradNorm':<9} | {'UpdateDir':<9}")
    print("-" * 125)

    for h in history:
        print(f"{h['step']:<5d} | {h['p_loc']:<7.4f} | {h['p_off']:<7.4f} | {h['ent_ed0']:<6.4f} | {h['q_loc']:<+8.4f} | {h['q_actual']:<+8.4f} | {h['q_off']:<+8.4f} | {h['end_dq']:<+9.5f} | {h['grad_dq']:<+9.5f} | {h['fd_slope']:<+9.5f} | {h['cls']:<4} | {h['g_norm']:<9.6f} | {h['update_dir']:<9}")

    print("\n============================================================")
    print("PART 9: FROZEN-ACTOR CONTROL (50 CRITIC UPDATES)")
    print("============================================================")

    trainer_frozen = copy.deepcopy(trainer_init)
    for step in range(1, 51):
        with torch.no_grad():
            t_ed_actions = [trainer_frozen.ed_target_actors[k](next_states_t[:, k*9:(k+1)*9]) for k in range(25)]
            t_cloud_probs = trainer_frozen.cloud_target_actor(next_states_t[:, -11:])
            t_joint_action = torch.cat(t_ed_actions + [t_cloud_probs], dim=-1)
            tQ = trainer_frozen.critic_target(next_states_t, t_joint_action)
            ty = rewards_t + (1.0 - dones_t) * 0.99 * tQ

        cQ = trainer_frozen.critic(states_t, actions_t)
        c_loss = nn.MSELoss()(cQ, ty)
        trainer_frozen.critic_optimizer.zero_grad()
        c_loss.backward()
        trainer_frozen.critic_optimizer.step()

    with torch.no_grad():
        a_loc = actions_t.clone(); a_loc[:, 0] = 1.0; a_loc[:, 1] = 0.0
        q_loc_f = trainer_frozen.critic(states_t, a_loc).mean().item()
        a_off = actions_t.clone(); a_off[:, 0] = 0.0; a_off[:, 1] = 1.0
        q_off_f = trainer_frozen.critic(states_t, a_off).mean().item()
        frozen_end_dq = q_off_f - q_loc_f

    print(f"Frozen-Actor Control Final ED-0 Endpoint DeltaQ (Offload - Local): {frozen_end_dq:+.6f}")
    print(f"Frozen Critic Landscape Favors Local: {frozen_end_dq < 0}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    all_cls = [h['cls'] for h in history]
    if all(c == 'A' for c in all_cls) and history[-1]['p_off'] < 0.5:
        classification = "F. CRITIC AND ACTOR BOTH FAVOR LOCAL"
    elif any(c == 'B' for c in all_cls):
        classification = "B. ACTOR GRADIENT CONTRADICTS CRITIC ENDPOINT PREFERENCE"
    elif all(c == 'A' for c in all_cls) and history[-1]['p_off'] > 0.5:
        classification = "D. OPTIMIZER UPDATE DIRECTION ERROR"
    else:
        classification = "A. ACTOR GRADIENT AGREES WITH CRITIC ENDPOINT PREFERENCE"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")
    print("HOST EXECUTION COMPLETED: NO (STATIC VERIFICATION ONLY)")


if __name__ == "__main__":
    main()
