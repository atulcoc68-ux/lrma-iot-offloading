import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.lrma_trainer import LRMATrainer


def get_stats(tensor):
    np_arr = tensor.detach().cpu().numpy()
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


def model_distance(model1, model2):
    dist_sq = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        dist_sq += (p1 - p2).norm(2).item() ** 2
    return dist_sq ** 0.5


def run_xi_experiment(xi_val, device_str):
    print(f"\n============================================================")
    print(f"RUNNING EXPERIMENT: xi_soft = {xi_val:.4f}")
    print(f"============================================================")

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
    next_states_t = torch.FloatTensor(next_states).to(trainer.device)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(trainer.device)
    dones_t = torch.FloatTensor(dones).unsqueeze(1).to(trainer.device)

    eval_ed0_state = states_t[0:1, 0:9]
    eval_cloud_state = states_t[0:1, -11:]

    max_c_dist = model_distance(trainer.critic, trainer.critic_target)
    history = []

    for step in range(1, 21):
        # 1. Target Q & TD Target prior to step
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

        # Measure Critic Gradient Norm
        cQ = trainer.critic(states_t, actions_t)
        loss_tmp = nn.MSELoss()(cQ, ty)
        trainer.critic_optimizer.zero_grad()
        loss_tmp.backward()

        c_grad_norm = 0.0
        for p in trainer.critic.parameters():
            if p.grad is not None:
                c_grad_norm += p.grad.norm(2).item() ** 2
        c_grad_norm = c_grad_norm ** 0.5
        trainer.critic_optimizer.zero_grad()

        # Input Gradient dQ/dA & ED-0 Action Sensitivity
        a_grad_in = actions_t.clone().detach().requires_grad_(True)
        q_for_grad = trainer.critic(states_t, a_grad_in)
        q_for_grad.sum().backward()
        dq_da = a_grad_in.grad.detach()
        dq_da_local = dq_da[:, 0].mean().item()
        dq_da_offload = dq_da[:, 1].mean().item()
        dq_da_diff = dq_da_offload - dq_da_local

        a_ed0_loc = actions_t.clone()
        a_ed0_loc[:, 0] = 1.0
        a_ed0_loc[:, 1] = 0.0

        a_ed0_off = actions_t.clone()
        a_ed0_off[:, 0] = 0.0
        a_ed0_off[:, 1] = 1.0

        with torch.no_grad():
            q_eval_loc = trainer.critic(states_t, a_ed0_loc).mean().item()
            q_eval_off = trainer.critic(states_t, a_ed0_off).mean().item()

        q_eval_diff = q_eval_off - q_eval_loc

        # Perform REAL train_step using exact xi_val
        a_loss, c_loss = trainer.train_step(batch_size=64, gamma=0.99, xi_soft=xi_val)

        c_dist = model_distance(trainer.critic, trainer.critic_target)
        if c_dist > max_c_dist:
            max_c_dist = c_dist

        # Inspect ED-0 & Cloud Policy States
        with torch.no_grad():
            ed0_p = trainer.ed_primary_actors[0](eval_ed0_state).squeeze(0)
            cloud_p = trainer.cloud_primary_actor(eval_cloud_state).squeeze(0)

        ed0_ent = compute_entropy(ed0_p)
        cloud_ent = compute_entropy(cloud_p)
        ed0_max_p = ed0_p.max().item()
        cloud_max_p = cloud_p.max().item()

        # Measure ED-0 Actor Gradient Norm
        trainer.ed_optimizers[0].zero_grad()
        p_probs = trainer.ed_primary_actors[0](states_t[:, 0:9])
        j_diff = actions_t.clone()
        j_diff[:, 0:2] = p_probs
        loss_act = -trainer.critic(states_t, j_diff).mean()
        loss_act.backward()

        ed0_grad_norm = 0.0
        for p in trainer.ed_primary_actors[0].parameters():
            if p.grad is not None:
                ed0_grad_norm += p.grad.norm(2).item() ** 2
        ed0_grad_norm = ed0_grad_norm ** 0.5
        trainer.ed_optimizers[0].zero_grad()

        scQ = get_stats(cQ)
        sty = get_stats(ty)

        history.append({
            'step': step,
            'cQ': scQ,
            'ty': sty,
            'c_loss': c_loss,
            'c_grad_norm': c_grad_norm,
            'c_dist': c_dist,
            'ed0_ent': ed0_ent,
            'ed0_max': ed0_max_p,
            'cloud_ent': cloud_ent,
            'cloud_max': cloud_max_p,
            'q_eval_diff': q_eval_diff,
            'dq_da_diff': dq_da_diff,
            'ed0_grad_norm': ed0_grad_norm
        })

        print(f"Step {step:02d} (xi={xi_val:.4f}) | Loss: {c_loss:7.4f} | Dist: {c_dist:7.4f} | Q-Max: {scQ['max']:7.4f}")
        print(f"  ED-0 Ent: {ed0_ent:.4f}, MaxP: {ed0_max_p:.4f} | Cloud Ent: {cloud_ent:.4f}, MaxP: {cloud_max_p:.4f}")
        print(f"  Q(off)-Q(loc): {q_eval_diff:+.6f} | dQ/dA Diff: {dq_da_diff:+.6f} | ED-0 Grad: {ed0_grad_norm:.6f}\n")

    finite_check = (
        torch.isfinite(ed0_p).all().item() and
        torch.isfinite(cloud_p).all().item() and
        all(torch.isfinite(p).all().item() for p in trainer.critic.parameters())
    )

    return {
        'xi_soft': xi_val,
        'final_dist': history[-1]['c_dist'],
        'max_dist': max_c_dist,
        'final_cQ_max': history[-1]['cQ']['max'],
        'final_loss': history[-1]['c_loss'],
        'final_ed0_ent': history[-1]['ed0_ent'],
        'final_ed0_max': history[-1]['ed0_max'],
        'final_cloud_ent': history[-1]['cloud_ent'],
        'final_cloud_max': history[-1]['cloud_max'],
        'final_q_eval_diff': history[-1]['q_eval_diff'],
        'final_dq_da_diff': history[-1]['dq_da_diff'],
        'final_ed0_grad': history[-1]['ed0_grad_norm'],
        'finite': finite_check,
        'history': history
    }


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

    xis = [0.001, 0.005, 0.01, 0.02, 0.05]
    results = []

    for xi in xis:
        res = run_xi_experiment(xi, device_str)
        results.append(res)

    print("\n============================================================")
    print("XI_SOFT POLICY STABILITY COMPARISON TABLE")
    print("============================================================")
    print(f"{'xi_soft':<8} | {'final_dist':<11} | {'max_dist':<10} | {'final_Qmax':<11} | {'final_loss':<11} | {'ED0_ent':<10} | {'ED0_maxP':<10} | {'cloud_ent':<10} | {'cloud_maxP':<10} | {'dQ/dA_diff':<11} | {'ED0_grad':<10}")
    print("-" * 140)

    for r in results:
        print(f"{r['xi_soft']:<8.4f} | {r['final_dist']:<11.4f} | {r['max_dist']:<10.4f} | {r['final_cQ_max']:<11.4f} | {r['final_loss']:<11.4f} | {r['final_ed0_ent']:<10.4f} | {r['final_ed0_max']:<10.4f} | {r['final_cloud_ent']:<10.4f} | {r['final_cloud_max']:<10.4f} | {r['final_dq_da_diff']:<+11.6f} | {r['final_ed0_grad']:<10.6f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION & RECOMMENDATION")
    print("============================================================")

    res_baseline = results[2]  # xi=0.01
    res_005 = results[4]     # xi=0.05
    res_001 = results[0]     # xi=0.001

    if res_005['final_dist'] < res_baseline['final_dist'] * 0.7 and res_005['final_ed0_ent'] > res_baseline['final_ed0_ent']:
        classification = "C. larger xi_soft is required"
    elif res_001['final_dist'] < res_baseline['final_dist'] * 0.7:
        classification = "B. smaller xi_soft is required"
    elif abs(res_005['final_ed0_ent'] - res_baseline['final_ed0_ent']) < 0.05 and res_005['final_ed0_max'] > 0.95:
        classification = "D. xi_soft has little effect; another mechanism dominates"
    elif res_baseline['final_ed0_ent'] > 0.3:
        classification = "A. xi_soft=0.01 is stable"
    else:
        classification = "E. no clear conclusion"

    print(f"CLASSIFICATION: {classification}")

    valid_results = [r for r in results if r['finite']]
    best_res = min(valid_results, key=lambda r: (r['final_dist'] + (1.0 - r['final_ed0_ent'])))
    print(f"BEST PERFORMING DIAGNOSTIC XI_SOFT: {best_res['xi_soft']:.4f} (Dist: {best_res['final_dist']:.4f}, ED0 Ent: {best_res['final_ed0_ent']:.4f})")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
