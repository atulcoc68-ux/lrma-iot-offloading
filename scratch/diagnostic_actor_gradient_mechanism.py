import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
from src.lrma_trainer import LRMATrainer
from src.lrma_networks import soft_update


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


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)

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

    history = []
    logit_growth_detected = False
    consistently_biased_grad = True
    prev_grad_sign = None

    print("============================================================")
    print("ACTOR GRADIENT MECHANISM DIAGNOSTIC")
    print("============================================================")

    for step in range(1, 21):
        # 1. Target Q & Critic TD update
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
        critic_loss = nn.MSELoss()(cQ, ty)

        trainer.critic_optimizer.zero_grad()
        critic_loss.backward()
        trainer.critic_optimizer.step()

        # 2. ED-0 Logits & Probabilities inspection before actor update
        ed0_primary_actor = trainer.ed_primary_actors[0]
        with torch.no_grad():
            # Get raw logits from layers 0..4 (before Softmax at layer 5)
            raw_logits = ed0_primary_actor.net[:5](states_t[:, 0:9])  # (64, 2)

            ed0_probs = torch.softmax(raw_logits, dim=-1)
            ed0_ent = compute_entropy(ed0_probs)
            ed0_max_p = ed0_probs.max().item()

            # Target ED-0 probabilities
            t_ed0_probs = trainer.ed_target_actors[0](states_t[:, 0:9])

        # 3. Action-Q comparisons:
        # Case A: Replay one-hot for all
        q_replay = trainer.critic(states_t, actions_t)

        # Case B: Current actor probability for ED-0
        a_ed0_prob = actions_t.clone()
        a_ed0_prob[:, 0:2] = ed0_probs
        q_ed0_prob = trainer.critic(states_t, a_ed0_prob)

        # Case C: Current target actor probability for ED-0
        a_ed0_target = actions_t.clone()
        a_ed0_target[:, 0:2] = t_ed0_probs
        q_ed0_target = trainer.critic(states_t, a_ed0_target)

        q_diff_prob_replay = (q_ed0_prob - q_replay).mean().item()
        q_diff_target_replay = (q_ed0_target - q_replay).mean().item()

        # 4. Input dQ/dA for ED-0 action slot
        a_grad_in = actions_t.clone().detach().requires_grad_(True)
        q_for_grad = trainer.critic(states_t, a_grad_in)
        q_for_grad.sum().backward()
        dq_da_ed0 = a_grad_in.grad[:, 0:2].detach()
        dq_da_norm = float(dq_da_ed0.norm(2).item())

        # 5. ED-0 Actor Loss & Gradient Calculation
        trainer.ed_optimizers[0].zero_grad()
        pred_probs = ed0_primary_actor(states_t[:, 0:9])
        joint_a_diff = actions_t.clone()
        joint_a_diff[:, 0:2] = pred_probs

        ed0_actor_loss = -trainer.critic(states_t, joint_a_diff).mean()
        ed0_actor_loss.backward()

        ed0_grad_norm = 0.0
        ed0_max_abs_grad = 0.0
        ed0_grad_vector = []
        for p in ed0_primary_actor.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                ed0_grad_norm += g.norm(2).item() ** 2
                max_abs = g.abs().max().item()
                if max_abs > ed0_max_abs_grad:
                    ed0_max_abs_grad = max_abs
                ed0_grad_vector.append(g.view(-1))

        ed0_grad_norm = ed0_grad_norm ** 0.5
        concat_grad = torch.cat(ed0_grad_vector)
        grad_sign = (concat_grad.mean() > 0).item()

        if prev_grad_sign is not None and prev_grad_sign != grad_sign:
            consistently_biased_grad = False
        prev_grad_sign = grad_sign

        trainer.ed_optimizers[0].step()

        # Update remaining actors & soft target updates
        for idx in range(1, trainer.num_ed):
            trainer.ed_optimizers[idx].zero_grad()
            p_probs = trainer.ed_primary_actors[idx](states_t[:, idx * 9 : (idx + 1) * 9])
            j_diff = actions_t.clone()
            j_diff[:, idx * 2 : (idx + 1) * 2] = p_probs
            loss_a = -trainer.critic(states_t, j_diff).mean()
            loss_a.backward()
            trainer.ed_optimizers[idx].step()

        trainer.cloud_optimizer.zero_grad()
        c_probs = trainer.cloud_primary_actor(states_t[:, -11:])
        j_c_diff = actions_t.clone()
        j_c_diff[:, -trainer.action_dim_cloud:] = c_probs
        loss_c = -trainer.critic(states_t, j_c_diff).mean()
        loss_c.backward()
        trainer.cloud_optimizer.step()

        for p, trg in zip(trainer.ed_primary_actors, trainer.ed_target_actors):
            soft_update(trg, p, 0.01)
        soft_update(trainer.cloud_target_actor, trainer.cloud_primary_actor, 0.01)
        soft_update(trainer.critic_target, trainer.critic, 0.01)

        logit_min = float(raw_logits.min().item())
        logit_max = float(raw_logits.max().item())
        logit_mean = float(raw_logits.mean().item())

        if abs(logit_max) > 10.0 or abs(logit_min) > 10.0:
            logit_growth_detected = True

        history.append({
            'step': step,
            'actor_loss': ed0_actor_loss.item(),
            'grad_norm': ed0_grad_norm,
            'max_abs_grad': ed0_max_abs_grad,
            'entropy': ed0_ent,
            'max_p': ed0_max_p,
            'q_actor_mean': q_ed0_prob.mean().item(),
            'q_actor_min': q_ed0_prob.min().item(),
            'q_actor_max': q_ed0_prob.max().item(),
            'dq_da_norm': dq_da_norm,
            'q_diff_prob': q_diff_prob_replay,
            'q_diff_target': q_diff_target_replay,
            'logit_min': logit_min,
            'logit_max': logit_max,
            'logit_mean': logit_mean
        })

        print(f"UPDATE {step:02d}")
        print(f"  Actor Loss: {ed0_actor_loss.item():7.4f} | Grad Norm: {ed0_grad_norm:7.6f} | Max Abs Grad: {ed0_max_abs_grad:7.6f}")
        print(f"  Entropy   : {ed0_ent:7.6f} | Max Prob : {ed0_max_p:7.6f} | dQ/dA Norm: {dq_da_norm:7.6f}")
        print(f"  Logits    : Min={logit_min:7.4f}, Max={logit_max:7.4f}, Mean={logit_mean:7.4f}")
        print(f"  Q Actor   : Min={q_ed0_prob.min().item():7.4f}, Max={q_ed0_prob.max().item():7.4f}, Mean={q_ed0_prob.mean().item():7.4f}")
        print(f"  Q Diff (Prob - Replay)  : {q_diff_prob_replay:+.6f}")
        print(f"  Q Diff (Target - Replay): {q_diff_target_replay:+.6f}\n")

    # Classification logic
    final_ent = history[-1]['entropy']
    final_max_p = history[-1]['max_p']
    q_diff_final = history[-1]['q_diff_target']

    if logit_growth_detected and final_max_p >= 0.98:
        classification = "C. LOGIT/OPTIMIZER DYNAMICS ARE DRIVING THE COLLAPSE"
    elif consistently_biased_grad and final_max_p >= 0.98:
        classification = "A. ACTOR GRADIENT DIRECTION IS STABLE AND CONSISTENTLY BIASED"
    elif abs(q_diff_final) > 0.5:
        classification = "B. CRITIC/TARGET DISAGREEMENT IS DRIVING THE ACTOR"
    else:
        classification = "D. NO CLEAR MECHANISM IDENTIFIED"

    print("============================================================")
    print("DIAGNOSTIC SUMMARY & CLASSIFICATION")
    print("============================================================")
    print(f"Initial Entropy -> Final Entropy    : {history[0]['entropy']:.6f} -> {final_ent:.6f}")
    print(f"Initial MaxProb -> Final MaxProb    : {history[0]['max_p']:.6f} -> {final_max_p:.6f}")
    print(f"Final Logit Range                   : [{history[-1]['logit_min']:.4f}, {history[-1]['logit_max']:.4f}]")
    print(f"Final dQ/dA Norm                    : {history[-1]['dq_da_norm']:.6f}")
    print(f"Final Q Diff (Target - Replay)      : {q_diff_final:+.6f}")
    print(f"\nFINAL CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
