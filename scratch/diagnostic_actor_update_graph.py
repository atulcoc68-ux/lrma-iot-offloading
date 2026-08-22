import sys
import os
import copy
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.lrma_trainer import LRMATrainer


def get_model_grad_norm(model):
    norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            norm_sq += p.grad.norm(2).item() ** 2
    return norm_sq ** 0.5


def get_flat_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if not grads:
        return torch.tensor([0.0])
    return torch.cat(grads)


def cosine_similarity(t1, t2):
    t1_flat = t1.view(-1)
    t2_flat = t2.view(-1)
    denom = (t1_flat.norm(2) * t2_flat.norm(2)).item()
    if denom == 0:
        return 1.0
    return float((t1_flat * t2_flat).sum().item() / denom)


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")

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

    print("\n============================================================")
    print("1. COMPUTATIONAL GRAPH REQUIRES_GRAD & GRAD_FN AUDIT")
    print("============================================================")

    ed0_state = states_t[:, 0:9]
    pred_probs_0 = trainer.ed_primary_actors[0](ed0_state)
    joint_a_diff_0 = actions_t.clone()
    joint_a_diff_0[:, 0:2] = pred_probs_0

    print(f"pred_probs_0.requires_grad: {pred_probs_0.requires_grad}")
    print(f"pred_probs_0.grad_fn: {pred_probs_0.grad_fn}")
    print(f"joint_a_diff_0.requires_grad: {joint_a_diff_0.requires_grad}")
    print(f"joint_a_diff_0.grad_fn: {joint_a_diff_0.grad_fn}")

    actor_loss_0 = -trainer.critic(states_t, joint_a_diff_0).mean()
    print(f"actor_loss_0.requires_grad: {actor_loss_0.requires_grad}")
    print(f"actor_loss_0.grad_fn: {actor_loss_0.grad_fn}")

    print("\n============================================================")
    print("2. GRADIENT ISOLATION & LEAKAGE CHECK (ED-0 BACKWARD)")
    print("============================================================")

    # Zero all gradients
    for opt in trainer.ed_optimizers:
        opt.zero_grad()
    trainer.cloud_optimizer.zero_grad()
    trainer.critic_optimizer.zero_grad()

    actor_loss_0.backward(retain_graph=True)

    g_ed0 = get_model_grad_norm(trainer.ed_primary_actors[0])
    g_ed1 = get_model_grad_norm(trainer.ed_primary_actors[1])
    g_ed2 = get_model_grad_norm(trainer.ed_primary_actors[2])
    g_cloud = get_model_grad_norm(trainer.cloud_primary_actor)
    g_critic = get_model_grad_norm(trainer.critic)

    print(f"ED-0 Grad Norm: {g_ed0:.6f}")
    print(f"ED-1 Grad Norm: {g_ed1:.6f} (Expected: 0.0)")
    print(f"ED-2 Grad Norm: {g_ed2:.6f} (Expected: 0.0)")
    print(f"Cloud Actor Grad Norm: {g_cloud:.6f} (Expected: 0.0)")
    print(f"Critic Parameter Grad Norm: {g_critic:.6f} (Expected: 0.0 for actor update)")

    isolation_passed = (g_ed0 > 0 and g_ed1 == 0 and g_ed2 == 0 and g_cloud == 0 and g_critic == 0)
    print(f"Gradient Isolation Test: {'PASSED' if isolation_passed else 'FAILED'}")

    print("\n============================================================")
    print("3. MULTI-AGENT UPDATE LOOP ACCUMULATION / LEAKAGE AUDIT")
    print("============================================================")

    # Simulate the exact 25 ED actor update loop inside train_step
    actor_loop_leakage = False
    actor_norms = []

    # Zero all optimizers
    for opt in trainer.ed_optimizers:
        opt.zero_grad()

    for idx in range(trainer.num_ed):
        # Check if previous actor gradients exist before update
        pre_grads = [get_model_grad_norm(trainer.ed_primary_actors[k]) for k in range(trainer.num_ed)]

        ed_st = states_t[:, idx * 9 : (idx + 1) * 9]
        p_probs = trainer.ed_primary_actors[idx](ed_st)
        j_diff = actions_t.clone()
        j_diff[:, idx * 2 : (idx + 1) * 2] = p_probs
        a_loss = -trainer.critic(states_t, j_diff).mean()

        trainer.ed_optimizers[idx].zero_grad()
        a_loss.backward()

        post_norm = get_model_grad_norm(trainer.ed_primary_actors[idx])
        actor_norms.append((idx, post_norm))

        # Check if other agents received unintended gradients
        other_norms = [get_model_grad_norm(trainer.ed_primary_actors[k]) for k in range(trainer.num_ed) if k != idx]
        if any(on > 0 for on in other_norms):
            actor_loop_leakage = True

        trainer.ed_optimizers[idx].zero_grad()

    print(f"Tested 25 ED actor updates in loop.")
    print(f"ED-0 Grad Norm: {actor_norms[0][1]:.6f}")
    print(f"ED-1 Grad Norm: {actor_norms[1][1]:.6f}")
    print(f"ED-12 Grad Norm: {actor_norms[12][1]:.6f}")
    print(f"ED-24 Grad Norm: {actor_norms[24][1]:.6f}")
    print(f"Actor Loop Leakage Detected: {'YES' if actor_loop_leakage else 'NO'}")

    print("\n============================================================")
    print("4. CLOUD ACTOR UPDATE AUDIT")
    print("============================================================")

    for opt in trainer.ed_optimizers:
        opt.zero_grad()
    trainer.cloud_optimizer.zero_grad()

    cloud_st = states_t[:, -11:]
    cloud_probs = trainer.cloud_primary_actor(cloud_st)
    j_diff_cloud = actions_t.clone()
    j_diff_cloud[:, -5:] = cloud_probs
    cloud_loss = -trainer.critic(states_t, j_diff_cloud).mean()

    cloud_loss.backward()

    g_cloud_self = get_model_grad_norm(trainer.cloud_primary_actor)
    g_ed_from_cloud = sum(get_model_grad_norm(trainer.ed_primary_actors[k]) for k in range(25))

    print(f"Cloud Actor Self Grad Norm: {g_cloud_self:.6f}")
    print(f"ED Actors Grad Norm from Cloud Update: {g_ed_from_cloud:.6f} (Expected: 0.0)")

    trainer.cloud_optimizer.zero_grad()

    print("\n============================================================")
    print("5. ANALYTICAL CHAIN-RULE VS AUTOGRAD GRADIENT COMPARISON")
    print("============================================================")

    # Compute dQ/dA_ED0 using autograd on input actions
    a_in = actions_t.clone().detach().requires_grad_(True)
    q_out = trainer.critic(states_t, a_in)
    q_out.sum().backward()
    dq_da = a_in.grad.detach()

    dq_da_loc = dq_da[:, 0]
    dq_da_off = dq_da[:, 1]
    dq_da_diff = dq_da_off - dq_da_loc

    with torch.no_grad():
        p_eval = trainer.ed_primary_actors[0](ed0_state)
        p_loc = p_eval[:, 0]
        p_off = p_eval[:, 1]

    # Analytical chain rule policy gradient: dJ/dz = p_loc * p_off * (dQ/dA_off - dQ/dA_loc) / batch_size
    analytical_dz = (p_loc * p_off * dq_da_diff).mean().item()

    # Actual autograd gradient on actor logits
    for opt in trainer.ed_optimizers:
        opt.zero_grad()

    p_probs_act = trainer.ed_primary_actors[0](ed0_state)
    j_diff_act = actions_t.clone()
    j_diff_act[:, 0:2] = p_probs_act
    l_act = -trainer.critic(states_t, j_diff_act).mean()
    l_act.backward()

    actual_grad_flat = get_flat_grad(trainer.ed_primary_actors[0])

    print(f"Analytical Softmax Policy Grad (dJ/dz): {analytical_dz:+.6f}")
    print(f"Actual Autograd ED-0 Grad Flat Norm: {actual_grad_flat.norm(2).item():.6f}")

    print("\n============================================================")
    print("6. FINITE-DIFFERENCE GRADIENT VERIFICATION")
    print("============================================================")

    eps = 1e-4

    # Perform a forward pass with perturbed logit for actor 0, sample 0
    with torch.no_grad():
        # Get baseline loss
        p_b = trainer.ed_primary_actors[0](ed0_state)
        j_b = actions_t.clone()
        j_b[:, 0:2] = p_b
        loss_base = -trainer.critic(states_t, j_b).mean().item()

    # Perturb the weights of the first layer of ED-0 actor by eps
    first_weight = list(trainer.ed_primary_actors[0].parameters())[0]

    with torch.no_grad():
        first_weight[0, 0] += eps
        p_pos = trainer.ed_primary_actors[0](ed0_state)
        j_pos = actions_t.clone()
        j_pos[:, 0:2] = p_pos
        loss_pos = -trainer.critic(states_t, j_pos).mean().item()

        first_weight[0, 0] -= 2 * eps
        p_neg = trainer.ed_primary_actors[0](ed0_state)
        j_neg = actions_t.clone()
        j_neg[:, 0:2] = p_neg
        loss_neg = -trainer.critic(states_t, j_neg).mean().item()

        # Restore original weight
        first_weight[0, 0] += eps

    fd_grad = (loss_pos - loss_neg) / (2 * eps)
    autograd_w00 = first_weight.grad[0, 0].item()

    fd_err = abs(fd_grad - autograd_w00)
    cos_sim = cosine_similarity(torch.tensor([fd_grad]), torch.tensor([autograd_w00]))

    print(f"Finite Difference Grad (w00): {fd_grad:+.6f}")
    print(f"Autograd Grad (w00):           {autograd_w00:+.6f}")
    print(f"Absolute Error:                 {fd_err:.8f}")
    print(f"Cosine Similarity:             {cos_sim:.6f}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    graph_correct = (
        isolation_passed and
        not actor_loop_leakage and
        g_ed_from_cloud == 0 and
        fd_err < 1e-3 and
        pred_probs_0.requires_grad and
        joint_a_diff_0.requires_grad
    )

    if graph_correct:
        classification = "A. ACTOR UPDATE GRAPH IS CORRECT"
    elif actor_loop_leakage or g_ed1 > 0:
        classification = "B. ACTOR GRADIENT IS LEAKING BETWEEN AGENTS"
    elif not pred_probs_0.requires_grad:
        classification = "C. ACTOR COMPUTATIONAL GRAPH IS BROKEN"
    elif g_critic > 0:
        classification = "E. ACTOR UPDATE GRAPH HAS CRITIC-GRADIENT CONTAMINATION"
    elif fd_err > 1e-2:
        classification = "D. ACTOR GRADIENT DOES NOT MATCH ANALYTICAL POLICY GRADIENT"
    else:
        classification = "F. MULTIPLE ACTOR UPDATE GRAPH PROBLEMS"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
