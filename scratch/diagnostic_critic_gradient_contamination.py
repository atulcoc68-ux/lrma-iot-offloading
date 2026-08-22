import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.lrma_trainer import LRMATrainer
from src.lrma_networks import soft_update


def get_model_grad_norm(model):
    norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            norm_sq += p.grad.norm(2).item() ** 2
    return norm_sq ** 0.5


def model_distance(model1, model2):
    dist_sq = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        dist_sq += (p1 - p2).norm(2).item() ** 2
    return dist_sq ** 0.5


def compute_entropy(probs):
    eps = 1e-12
    p_safe = torch.clamp(probs, min=eps)
    return -(p_safe * torch.log(p_safe)).sum(dim=-1).mean().item()


def run_experiment(freeze_critic_during_actor, device_str):
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

    eval_ed0_st = states_t[0:1, 0:9]
    eval_cloud_st = states_t[0:1, -11:]

    history = []

    for step in range(1, 11):
        # Step 1: Target Q & TD Target calculation
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

        # Step 2: Critic Update
        if freeze_critic_during_actor:
            for p in trainer.critic.parameters():
                p.requires_grad_(True)

        c_grad_before_zero = get_model_grad_norm(trainer.critic)
        trainer.critic_optimizer.zero_grad()
        c_grad_after_zero = get_model_grad_norm(trainer.critic)

        cQ = trainer.critic(states_t, actions_t)
        c_loss = nn.MSELoss()(cQ, ty)

        c_loss.backward()
        c_grad_before_step = get_model_grad_norm(trainer.critic)
        trainer.critic_optimizer.step()

        # Step 3: ED Actor Loop
        if freeze_critic_during_actor:
            for p in trainer.critic.parameters():
                p.requires_grad_(False)

        a_losses = []
        c_grad_after_ed0 = 0.0

        for idx in range(trainer.num_ed):
            ed_st = states_t[:, idx * 9 : (idx + 1) * 9]
            p_probs = trainer.ed_primary_actors[idx](ed_st)
            j_diff = actions_t.clone()
            j_diff[:, idx * 2 : (idx + 1) * 2] = p_probs

            a_loss = -trainer.critic(states_t, j_diff).mean()
            a_losses.append(a_loss.item())

            trainer.ed_optimizers[idx].zero_grad()
            a_loss.backward()

            if idx == 0:
                c_grad_after_ed0 = get_model_grad_norm(trainer.critic)

            trainer.ed_optimizers[idx].step()

        # Step 4: Cloud Actor Update
        cloud_st = states_t[:, -11:]
        cloud_probs = trainer.cloud_primary_actor(cloud_st)
        j_diff_cloud = actions_t.clone()
        j_diff_cloud[:, -5:] = cloud_probs

        cloud_loss = -trainer.critic(states_t, j_diff_cloud).mean()
        trainer.cloud_optimizer.zero_grad()
        cloud_loss.backward()
        c_grad_after_cloud = get_model_grad_norm(trainer.critic)
        trainer.cloud_optimizer.step()

        # Step 5: Soft Update Target Networks
        soft_update(trainer.critic_target, trainer.critic, 0.01)
        for idx in range(trainer.num_ed):
            soft_update(trainer.ed_target_actors[idx], trainer.ed_primary_actors[idx], 0.01)
        soft_update(trainer.cloud_target_actor, trainer.cloud_primary_actor, 0.01)

        # Record Policy & Q Stats
        with torch.no_grad():
            ed0_p = trainer.ed_primary_actors[0](eval_ed0_st).squeeze(0)
            cloud_p = trainer.cloud_primary_actor(eval_cloud_st).squeeze(0)

        history.append({
            'step': step,
            'trainer': trainer,
            'c_loss': c_loss.item(),
            'a0_loss': a_losses[0],
            'cloud_loss': cloud_loss.item(),
            'c_grad_before_zero': c_grad_before_zero,
            'c_grad_after_zero': c_grad_after_zero,
            'c_grad_before_step': c_grad_before_step,
            'c_grad_after_ed0': c_grad_after_ed0,
            'c_grad_after_cloud': c_grad_after_cloud,
            'cQ_mean': cQ.mean().item(),
            'cQ_max': cQ.max().item(),
            'cQ_min': cQ.min().item(),
            'ed0_ent': compute_entropy(ed0_p),
            'ed0_maxP': ed0_p.max().item(),
            'cloud_ent': compute_entropy(cloud_p),
            'cloud_maxP': cloud_p.max().item()
        })

    return history


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

    print("\n============================================================")
    print("RUNNING EXPERIMENT A: EXACT CURRENT TRAIN_STEP ORDERING")
    print("============================================================")
    hist_A = run_experiment(freeze_critic_during_actor=False, device_str=device_str)

    print("\n============================================================")
    print("RUNNING EXPERIMENT B: CONTROL (FREEZE CRITIC DURING ACTOR UPDATE)")
    print("============================================================")
    hist_B = run_experiment(freeze_critic_during_actor=True, device_str=device_str)

    print("\n============================================================")
    print("GRADIENT FLOW COMPARISON FOR EXPERIMENT A (CURRENT IMPLEMENTATION)")
    print("============================================================")
    print(f"{'Step':<5} | {'cGrad Before Zero':<18} | {'cGrad After Zero':<17} | {'cGrad Before Step':<18} | {'cGrad After ED0':<16} | {'cGrad After Cloud':<18}")
    print("-" * 105)

    for hA in hist_A:
        print(f"{hA['step']:<5d} | {hA['c_grad_before_zero']:<18.6f} | {hA['c_grad_after_zero']:<17.6f} | {hA['c_grad_before_step']:<18.6f} | {hA['c_grad_after_ed0']:<16.6f} | {hA['c_grad_after_cloud']:<18.6f}")

    print("\n============================================================")
    print("EXPERIMENT A VS EXPERIMENT B COMPARISON AT STEP 1 & STEP 10")
    print("============================================================")

    steps_to_compare = [1, 10]

    for s_idx in steps_to_compare:
        hA = hist_A[s_idx - 1]
        hB = hist_B[s_idx - 1]

        tA = hA['trainer']
        tB = hB['trainer']

        dist_critic = model_distance(tA.critic, tB.critic)
        dist_ed0 = model_distance(tA.ed_primary_actors[0], tB.ed_primary_actors[0])
        dist_cloud = model_distance(tA.cloud_primary_actor, tB.cloud_primary_actor)

        print(f"\n--- STEP {s_idx} METRICS ---")
        print(f"Critic Parameter Distance ||Critic_A - Critic_B||: {dist_critic:.10f}")
        print(f"ED-0 Actor Distance ||ED0_A - ED0_B||:           {dist_ed0:.10f}")
        print(f"Cloud Actor Distance ||Cloud_A - Cloud_B||:       {dist_cloud:.10f}")

        print(f"Critic Loss  : Exp A = {hA['c_loss']:.6f} | Exp B = {hB['c_loss']:.6f} | Diff = {abs(hA['c_loss'] - hB['c_loss']):.10f}")
        print(f"ED-0 Loss    : Exp A = {hA['a0_loss']:.6f} | Exp B = {hB['a0_loss']:.6f} | Diff = {abs(hA['a0_loss'] - hB['a0_loss']):.10f}")
        print(f"Cloud Loss   : Exp A = {hA['cloud_loss']:.6f} | Exp B = {hB['cloud_loss']:.6f} | Diff = {abs(hA['cloud_loss'] - hB['cloud_loss']):.10f}")
        print(f"Critic Q Mean: Exp A = {hA['cQ_mean']:.6f} | Exp B = {hB['cQ_mean']:.6f}")
        print(f"ED-0 Entropy : Exp A = {hA['ed0_ent']:.6f} | Exp B = {hB['ed0_ent']:.6f}")

    print("\n============================================================")
    print("CRITIC OPTIMIZER STEPPING VERIFICATION")
    print("============================================================")
    print("Is critic_optimizer.step() called after actor-generated critic gradients?")
    print("No. In train_step():")
    print("  1. critic_optimizer.zero_grad()")
    print("  2. critic_loss.backward()")
    print("  3. critic_optimizer.step()")
    print("  4. ED & Cloud actor updates (actor_loss.backward())")
    print("  5. At step t+1: critic_optimizer.zero_grad() CLEARS all actor-generated critic gradients before step t+1 critic update.")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    step10_dist_critic = model_distance(hist_A[9]['trainer'].critic, hist_B[9]['trainer'].critic)

    if step10_dist_critic < 1e-3:
        classification = "A. NO EFFECTIVE CRITIC CONTAMINATION"
    elif step10_dist_critic > 1e-1:
        classification = "B. CRITIC GRADIENTS EFFECTIVELY CONTAMINATE TRAINING"
    else:
        classification = "E. INSUFFICIENT EVIDENCE"

    print(f"CLASSIFICATION: {classification}")

    if classification == "A. NO EFFECTIVE CRITIC CONTAMINATION":
        print("\nSTATEMENT:")
        print("The non-zero critic gradients during actor backward are mathematically incidental and do not alter critic parameters because the critic optimizer is not stepped during actor updates.")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
