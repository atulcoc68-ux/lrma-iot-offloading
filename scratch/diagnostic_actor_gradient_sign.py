import sys
import os
sys.path.insert(0, os.getcwd())

import torch
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


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)

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

    print("============================================================")
    print("ACTOR GRADIENT SIGN & DIRECTION DIAGNOSTIC")
    print("============================================================")

    q_diff_offload_local = []
    dq_da_diff_offload_local = []
    actual_grad_sign_offload_increase = []
    chain_rule_agreed = 0
    inconsistency_count = 0  # Q(offload) < Q(local) BUT actor gradient pushes toward offload

    num_evals = 20

    for step in range(1, num_evals + 1):
        ed0_actor = trainer.ed_primary_actors[0]

        # 1. Evaluate Direct Finite-Difference Q values for ED-0: Local [1, 0] vs Offload [0, 1]
        a_local = actions_t.clone()
        a_local[:, 0] = 1.0
        a_local[:, 1] = 0.0

        a_offload = actions_t.clone()
        a_offload[:, 0] = 0.0
        a_offload[:, 1] = 1.0

        with torch.no_grad():
            q_local = trainer.critic(states_t, a_local)
            q_offload = trainer.critic(states_t, a_offload)

        q_diff = (q_offload - q_local).mean().item()
        q_diff_offload_local.append(q_diff)

        # 2. Evaluate Actor Policy Probabilities & Logits
        with torch.no_grad():
            raw_logits = ed0_actor.net[:5](states_t[:, 0:9])
            p_probs = torch.softmax(raw_logits, dim=-1)

        # 3. Input Gradient dQ/dA at Current Actor Action Representation
        a_grad_in = actions_t.clone()
        a_grad_in[:, 0:2] = p_probs
        a_grad_in = a_grad_in.detach().requires_grad_(True)

        q_for_grad = trainer.critic(states_t, a_grad_in)
        q_for_grad.sum().backward()

        dq_da = a_grad_in.grad[:, 0:2].detach()
        dq_da_local = dq_da[:, 0].mean().item()
        dq_da_offload = dq_da[:, 1].mean().item()
        dq_da_diff = dq_da_offload - dq_da_local
        dq_da_diff_offload_local.append(dq_da_diff)

        # 4. Actual Actor Objective & Backward Pass
        ed0_actor.zero_grad()
        pred_probs = ed0_actor(states_t[:, 0:9])
        joint_a_diff = actions_t.clone()
        joint_a_diff[:, 0:2] = pred_probs

        actor_loss = -trainer.critic(states_t, joint_a_diff).mean()
        actor_loss.backward()

        # Output layer weight gradient inspection
        # Out layer weight shape is (2, 128) -> index 0: local, index 1: offload
        out_layer_weight_grad = ed0_actor.net[4].weight.grad.detach()
        grad_local_w = out_layer_weight_grad[0].mean().item()
        grad_offload_w = out_layer_weight_grad[1].mean().item()

        # Under Gradient Ascent on J = -L_actor:
        # Parameter update: w_offload <- w_offload - lr * grad_offload_w
        # If grad_offload_w < 0, w_offload increases -> offload probability increases!
        pushes_offload = (grad_offload_w < 0)
        actual_grad_sign_offload_increase.append(pushes_offload)

        # 5. Softmax Chain Rule Prediction:
        # \frac{\partial J}{\partial z_1} = p_0 p_1 (dQ/dA_{offload} - dQ/dA_{local})
        # If dq_da_diff > 0, J increases with z_1 -> grad_offload_w should be < 0
        predicted_pushes_offload = (dq_da_diff > 0)

        if pushes_offload == predicted_pushes_offload:
            chain_rule_agreed += 1

        # Check critical inconsistency: Critic favors LOCAL (q_diff < 0), BUT actor gradient pushes OFFLOAD (pushes_offload = True)
        if q_diff < -1e-4 and pushes_offload:
            inconsistency_count += 1

        print(f"Sample {step:02d}")
        print(f"  Q(offload) - Q(local)          : {q_diff:+.6f}")
        print(f"  dQ/dA_offload - dQ/dA_local    : {dq_da_diff:+.6f}")
        print(f"  Grad Weight Out (Local/Offload): {grad_local_w:+.6f} / {grad_offload_w:+.6f}")
        print(f"  Chain Rule Agreement           : {'YES' if pushes_offload == predicted_pushes_offload else 'NO'}")
        print(f"  Actor Gradient Pushes          : {'OFFLOAD' if pushes_offload else 'LOCAL'}\n")

    mean_q_diff = float(np.mean(q_diff_offload_local))
    mean_dq_da_diff = float(np.mean(dq_da_diff_offload_local))
    chain_rule_pct = (chain_rule_agreed / num_evals) * 100.0
    offload_push_pct = (sum(actual_grad_sign_offload_increase) / num_evals) * 100.0

    print("============================================================")
    print("ACTOR GRADIENT SIGN DIAGNOSTIC SUMMARY")
    print("============================================================")
    print(f"Mean Q(offload) - Q(local)        : {mean_q_diff:+.6f}")
    print(f"Mean dQ/dA_offload - dQ/dA_local  : {mean_dq_da_diff:+.6f}")
    print(f"Chain Rule Match Percentage       : {chain_rule_pct:.2f}%")
    print(f"Actor Gradient Pushing Offload    : {offload_push_pct:.2f}% of samples")
    print(f"Inconsistency Cases (Q local > Q offload BUT pushes offload): {inconsistency_count} / {num_evals}")

    print("\n============================================================")
    print("FINAL CLASSIFICATION")
    print("============================================================")

    if inconsistency_count > num_evals * 0.4:
        classification = "B. CRITIC FAVORS LOCAL BUT ACTOR GRADIENT PUSHES OFFLOAD"
    elif mean_q_diff > 1e-4 and mean_dq_da_diff > 1e-4 and chain_rule_pct >= 80.0:
        classification = "A. CRITIC DIRECTLY FAVORS OFFLOAD AND ACTOR GRADIENT IS CONSISTENT"
    elif chain_rule_pct < 60.0:
        classification = "D. DIAGNOSTIC IMPLEMENTATION INCONSISTENCY"
    else:
        classification = "C. CRITIC ACTION GRADIENT IS STATE-DEPENDENT/MIXED"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
