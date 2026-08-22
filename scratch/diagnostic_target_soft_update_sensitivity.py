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


def run_experiment(xi_soft_val, device_str):
    print(f"\n============================================================")
    print(f"STARTING EXPERIMENT: xi_soft = {xi_soft_val}")
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

    eval_ed_state = states_t[0:1, 0:9]
    eval_cloud_state = states_t[0:1, -11:]

    init_c_dist = model_distance(trainer.critic, trainer.critic_target)

    with torch.no_grad():
        cQ_init = trainer.critic(states_t, actions_t)
        t_ed_a_init = []
        for idx in range(trainer.num_ed):
            ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
            t_ed_a_init.append(trainer.ed_target_actors[idx](ed_st))
        cloud_st = next_states_t[:, -11:]
        t_cloud_probs_init = trainer.cloud_target_actor(cloud_st)
        t_joint_a_init = torch.cat(t_ed_a_init + [t_cloud_probs_init], dim=-1)

        tQ_init = trainer.critic_target(next_states_t, t_joint_a_init)

    init_cQ_stats = get_stats(cQ_init)
    init_tQ_stats = get_stats(tQ_init)

    history = []
    max_c_dist = init_c_dist
    max_grad_norm = 0.0

    for step in range(1, 21):
        # Measure critic gradient norm prior to train_step update
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
        loss_critic_tmp = nn.MSELoss()(cQ, ty)
        
        trainer.critic_optimizer.zero_grad()
        loss_critic_tmp.backward()
        
        c_grad_norm = 0.0
        for p in trainer.critic.parameters():
            if p.grad is not None:
                c_grad_norm += p.grad.norm(2).item() ** 2
        c_grad_norm = c_grad_norm ** 0.5
        trainer.critic_optimizer.zero_grad()

        # Perform exact train_step update
        a_loss, c_loss = trainer.train_step(batch_size=64, gamma=0.99, xi_soft=xi_soft_val)

        with torch.no_grad():
            cQ_post = trainer.critic(states_t, actions_t)
            t_ed_a_post = []
            for idx in range(trainer.num_ed):
                ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                t_ed_a_post.append(trainer.ed_target_actors[idx](ed_st))
            cloud_st = next_states_t[:, -11:]
            t_cloud_probs_post = trainer.cloud_target_actor(cloud_st)
            t_joint_a_post = torch.cat(t_ed_a_post + [t_cloud_probs_post], dim=-1)

            tQ_post = trainer.critic_target(next_states_t, t_joint_a_post)
            ty_post = rewards_t + (1.0 - dones_t) * 0.99 * tQ_post

            ed0_p = trainer.ed_primary_actors[0](eval_ed_state).squeeze(0)
            cloud_p = trainer.cloud_primary_actor(eval_cloud_state).squeeze(0)

        ed0_ent = compute_entropy(ed0_p)
        cloud_ent = compute_entropy(cloud_p)
        ed0_max_p = ed0_p.max().item()
        cloud_max_p = cloud_p.max().item()

        c_dist = model_distance(trainer.critic, trainer.critic_target)
        if c_dist > max_c_dist:
            max_c_dist = c_dist
        if c_grad_norm > max_grad_norm:
            max_grad_norm = c_grad_norm

        scQ = get_stats(cQ_post)
        stQ = get_stats(tQ_post)
        sty = get_stats(ty_post)

        history.append({
            'step': step,
            'cQ': scQ,
            'tQ': stQ,
            'ty': sty,
            'c_loss': c_loss,
            'c_grad_norm': c_grad_norm,
            'c_dist': c_dist,
            'ed0_ent': ed0_ent,
            'ed0_max': ed0_max_p,
            'cloud_ent': cloud_ent,
            'cloud_max': cloud_max_p
        })

        print(f"Update {step:02d} (xi={xi_soft_val:.2f}) | Critic Loss: {c_loss:7.4f} | Grad Norm: {c_grad_norm:7.4f} | Dist: {c_dist:7.4f}")
        print(f"  Current Q: min={scQ['min']:7.4f}, max={scQ['max']:7.4f}, mean={scQ['mean']:7.4f}, std={scQ['std']:7.4f}")
        print(f"  Target Q : min={stQ['min']:7.4f}, max={stQ['max']:7.4f}, mean={stQ['mean']:7.4f}, std={stQ['std']:7.4f}")
        print(f"  Target Y : min={sty['min']:7.4f}, max={sty['max']:7.4f}, mean={sty['mean']:7.4f}, std={sty['std']:7.4f}")
        print(f"  ED-0 Ent: {ed0_ent:.4f}, MaxP: {ed0_max_p:.4f} | Cloud Ent: {cloud_ent:.4f}, MaxP: {cloud_max_p:.4f}\n")

    # Validity checks
    critic_finite = all(torch.isfinite(p).all().item() for p in trainer.critic.parameters())
    target_critic_finite = all(torch.isfinite(p).all().item() for p in trainer.critic_target.parameters())
    actor_finite = all(all(torch.isfinite(p).all().item() for p in a.parameters()) for a in trainer.ed_primary_actors) and all(torch.isfinite(p).all().item() for p in trainer.cloud_primary_actor.parameters())
    target_actor_finite = all(all(torch.isfinite(p).all().item() for p in a.parameters()) for a in trainer.ed_target_actors) and all(torch.isfinite(p).all().item() for p in trainer.cloud_target_actor.parameters())

    probs_finite = torch.isfinite(ed0_p).all().item() and torch.isfinite(cloud_p).all().item()
    ed_sum_valid = torch.allclose(ed0_p.sum(), torch.tensor(1.0, device=trainer.device))
    cloud_sum_valid = torch.allclose(cloud_p.sum(), torch.tensor(1.0, device=trainer.device))

    all_valid = (critic_finite and target_critic_finite and actor_finite and target_actor_finite and probs_finite and ed_sum_valid and cloud_sum_valid)

    print(f"--- SUMMARY EXPERIMENT xi_soft = {xi_soft_val} ---")
    print(f"Initial critic-target distance: {init_c_dist:.6f}")
    print(f"Final critic-target distance  : {history[-1]['c_dist']:.6f}")
    print(f"Maximum critic-target distance: {max_c_dist:.6f}")
    print(f"Initial critic Q range        : [{init_cQ_stats['min']:.4f}, {init_cQ_stats['max']:.4f}]")
    print(f"Final critic Q range          : [{history[-1]['cQ']['min']:.4f}, {history[-1]['cQ']['max']:.4f}]")
    print(f"Initial target Q range        : [{init_tQ_stats['min']:.4f}, {init_tQ_stats['max']:.4f}]")
    print(f"Final target Q range          : [{history[-1]['tQ']['min']:.4f}, {history[-1]['tQ']['max']:.4f}]")
    print(f"Maximum critic gradient norm  : {max_grad_norm:.6f}")
    print(f"Final ED-0 entropy            : {history[-1]['ed0_ent']:.6f}")
    print(f"Final Cloud entropy           : {history[-1]['cloud_ent']:.6f}")
    print(f"Final ED-0 maximum probability: {history[-1]['ed0_max']:.6f}")
    print(f"Final Cloud maximum probability: {history[-1]['cloud_max']:.6f}")
    print(f"All Validity Checks Passed    : {'YES' if all_valid else 'NO'}")

    return {
        'xi_soft': xi_soft_val,
        'final_dist': history[-1]['c_dist'],
        'max_dist': max_c_dist,
        'final_c_loss': history[-1]['c_loss'],
        'final_ed_ent': history[-1]['ed0_ent'],
        'final_ed_max': history[-1]['ed0_max'],
        'final_cloud_ent': history[-1]['cloud_ent'],
        'final_cloud_max': history[-1]['cloud_max'],
        'max_grad_norm': max_grad_norm,
        'final_cQ_max': history[-1]['cQ']['max']
    }


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

    res_001 = run_experiment(0.01, device_str)
    res_005 = run_experiment(0.05, device_str)
    res_010 = run_experiment(0.10, device_str)

    print("\n============================================================")
    print("TARGET SOFT UPDATE SENSITIVITY COMPARISON TABLE")
    print("============================================================")
    print(f"{'xi_soft':<8} | {'final_dist':<19} | {'max_dist':<16} | {'final_loss':<16} | {'final_ED_ent':<16} | {'final_ED_max':<16} | {'cloud_ent':<17} | {'cloud_max':<15}")
    print("-" * 140)

    for r in [res_001, res_005, res_010]:
        print(f"{r['xi_soft']:<8.2f} | {r['final_dist']:<19.6f} | {r['max_dist']:<16.6f} | {r['final_c_loss']:<16.6f} | {r['final_ed_ent']:<16.6f} | {r['final_ed_max']:<16.6f} | {r['final_cloud_ent']:<17.6f} | {r['final_cloud_max']:<15.6f}")

    print("\n============================================================")
    print("FINAL DIAGNOSTIC CLASSIFICATION")
    print("============================================================")

    d_001 = res_001['final_dist']
    d_005 = res_005['final_dist']
    d_010 = res_010['final_dist']

    if d_010 < d_001 * 0.5 and res_010['final_cQ_max'] < res_001['final_cQ_max']:
        classification = "C. Target lag strongly supported as primary instability mechanism"
    elif d_010 < d_001:
        classification = "B. Target lag likely contributes"
    elif res_010['final_c_loss'] > res_001['final_c_loss'] * 2.0:
        classification = "D. Increasing xi_soft causes instability"
    else:
        classification = "A. Target lag NOT significant"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("PERSISTENT RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
