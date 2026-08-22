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
    print("PART 1: CURRENT IMPLEMENTATION HYBRID OBJECTIVE EVALUATION")
    print("============================================================")

    q_replay_mean = trainer.critic(states_t, actions_t).mean().item()

    hybrid_results = []
    for ed_idx in range(25):
        ed_st = states_t[:, ed_idx * 9 : (ed_idx + 1) * 9]
        p_probs = trainer.ed_primary_actors[ed_idx](ed_st)
        j_diff = actions_t.clone()
        j_diff[:, ed_idx * 2 : (ed_idx + 1) * 2] = p_probs

        q_hyb = trainer.critic(states_t, j_diff)
        loss_hyb = -q_hyb.mean()

        trainer.ed_optimizers[ed_idx].zero_grad()
        loss_hyb.backward()

        g_flat = get_flat_grad(trainer.ed_primary_actors[ed_idx])
        g_norm = float(g_flat.norm(2).item())

        hybrid_results.append({
            'ed_idx': ed_idx,
            'q_mean': q_hyb.mean().item(),
            'q_std': q_hyb.std().item(),
            'loss': loss_hyb.item(),
            'grad_norm': g_norm,
            'grad_flat': g_flat.clone(),
            'j_diff': j_diff
        })
        trainer.ed_optimizers[ed_idx].zero_grad()

    print(f"{'ED Index':<10} | {'Mean Q':<12} | {'Std Q':<12} | {'Actor Loss':<12} | {'Grad Norm':<12}")
    print("-" * 65)
    for hr in hybrid_results[:5]:
        print(f"{hr['ed_idx']:<10d} | {hr['q_mean']:<+12.6f} | {hr['q_std']:<12.6f} | {hr['loss']:<+12.6f} | {hr['grad_norm']:<12.6f}")

    print("\n============================================================")
    print("PART 2: FULL CURRENT-POLICY JOINT ACTION EVALUATION")
    print("============================================================")

    curr_ed_probs = []
    for ed_idx in range(25):
        ed_st = states_t[:, ed_idx * 9 : (ed_idx + 1) * 9]
        curr_ed_probs.append(trainer.ed_primary_actors[ed_idx](ed_st))

    cloud_st = states_t[:, -11:]
    curr_cloud_probs = trainer.cloud_primary_actor(cloud_st)

    a_full_policy = torch.cat(curr_ed_probs + [curr_cloud_probs], dim=-1)

    print(f"a_full_policy shape: {a_full_policy.shape}")
    print(f"All finite: {torch.isfinite(a_full_policy).all().item()}")

    ed_sum_check = all(torch.allclose(curr_ed_probs[k].sum(dim=-1), torch.ones(batch_size, device=trainer.device)) for k in range(25))
    cloud_sum_check = torch.allclose(curr_cloud_probs.sum(dim=-1), torch.ones(batch_size, device=trainer.device))

    print(f"ED probability pairs sum to 1: {ed_sum_check}")
    print(f"Cloud probability vector sums to 1: {cloud_sum_check}")

    q_full = trainer.critic(states_t, a_full_policy)
    q_full_stats = get_stats(q_full)

    print(f"Q_current_policy: Mean = {q_full_stats['mean']:+.6f}, Std = {q_full_stats['std']:.6f}, Min = {q_full_stats['min']:+.6f}, Max = {q_full_stats['max']:+.6f}")

    print("\n============================================================")
    print("PART 3: DISCRETE REPLAY VS CURRENT POLICY Q COMPARISON")
    print("============================================================")

    print(f"Q_replay Mean: {q_replay_mean:+.6f}")
    print(f"Q_hybrid_ED0 - Q_replay: {hybrid_results[0]['q_mean'] - q_replay_mean:+.6f}")
    print(f"Q_current_policy - Q_replay: {q_full_stats['mean'] - q_replay_mean:+.6f}")

    print("\n============================================================")
    print("PART 4: POLICY GRADIENT DIRECTION COMPARISON (HYBRID VS FULL)")
    print("============================================================")

    full_results = []
    loss_full_total = -q_full.mean()

    for ed_idx in range(25):
        trainer.ed_optimizers[ed_idx].zero_grad()

    loss_full_total.backward(retain_graph=True)

    cos_sims = []
    for ed_idx in range(25):
        g_full = get_flat_grad(trainer.ed_primary_actors[ed_idx])
        g_hyb = hybrid_results[ed_idx]['grad_flat']

        sim = cosine_similarity(g_hyb, g_full)
        cos_sims.append(sim)

        full_results.append({
            'ed_idx': ed_idx,
            'grad_norm': float(g_full.norm(2).item()),
            'grad_flat': g_full.clone(),
            'cos_sim': sim
        })
        trainer.ed_optimizers[ed_idx].zero_grad()

    print(f"{'ED Index':<10} | {'Hybrid Grad Norm':<18} | {'Full Grad Norm':<16} | {'Cosine Sim':<12} | {'Classification':<15}")
    print("-" * 80)
    for i in range(5):
        sim_val = cos_sims[i]
        if sim_val > 0.8:
            cls_str = "SAME DIRECTION"
        elif sim_val < -0.8:
            cls_str = "DIFFERENT DIRECTION"
        else:
            cls_str = "NEAR ORTHOGONAL"
        print(f"{i:<10d} | {hybrid_results[i]['grad_norm']:<18.6f} | {full_results[i]['grad_norm']:<16.6f} | {sim_val:<+12.6f} | {cls_str:<15}")

    avg_sim = float(np.mean(cos_sims))
    print(f"... (Average Cosine Similarity across all 25 EDs: {avg_sim:+.6f})")

    print("\n============================================================")
    print("PART 5: OFFLOAD VS LOCAL LOGIT GRADIENT BIAS")
    print("============================================================")

    offload_push_count = 0
    local_push_count = 0

    for ed_idx in range(25):
        a_in = actions_t.clone().detach().requires_grad_(True)
        q_val = trainer.critic(states_t, a_in)
        q_val.sum().backward()

        dq_da = a_in.grad.detach()
        dq_loc = dq_da[:, ed_idx*2].mean().item()
        dq_off = dq_da[:, ed_idx*2+1].mean().item()
        dq_diff = dq_off - dq_loc

        pushing_offload = (dq_diff > 0)
        if pushing_offload:
            offload_push_count += 1
        else:
            local_push_count += 1

    pct_offload = (offload_push_count / 25) * 100.0
    pct_local = (local_push_count / 25) * 100.0

    print(f"Percentage of ED Actors Pushing Toward OFFLOAD: {pct_offload:.1f}% ({offload_push_count}/25)")
    print(f"Percentage of ED Actors Pushing Toward LOCAL  : {pct_local:.1f}% ({local_push_count}/25)")

    print("\n============================================================")
    print("PART 6 & 7: 20-STEP SEQUENTIAL SIMULATION (HYBRID VS FULL)")
    print("============================================================")

    sim_trainer_A = LRMATrainer(num_ed=25, num_mes=5, device=device_str)
    sim_trainer_A.critic.load_state_dict(trainer.critic.state_dict())
    for k in range(25):
        sim_trainer_A.ed_primary_actors[k].load_state_dict(trainer.ed_primary_actors[k].state_dict())
    sim_trainer_A.cloud_primary_actor.load_state_dict(trainer.cloud_primary_actor.state_dict())

    eval_ed0_st = states_t[0:1, 0:9]
    eval_cloud_st = states_t[0:1, -11:]

    with torch.no_grad():
        ed0_p_init = sim_trainer_A.ed_primary_actors[0](eval_ed0_st).squeeze(0)
        cloud_p_init = sim_trainer_A.cloud_primary_actor(eval_cloud_st).squeeze(0)

    ed0_ent_start = compute_entropy(ed0_p_init)
    ed0_max_start = ed0_p_init.max().item()
    cloud_ent_start = compute_entropy(cloud_p_init)
    cloud_max_start = cloud_p_init.max().item()

    hybrid_grad_norms = []
    for step in range(1, 21):
        step_g_norms = []
        for idx in range(25):
            ed_st = states_t[:, idx * 9 : (idx + 1) * 9]
            p_probs = sim_trainer_A.ed_primary_actors[idx](ed_st)
            j_diff = actions_t.clone()
            j_diff[:, idx * 2 : (idx + 1) * 2] = p_probs

            a_loss = -sim_trainer_A.critic(states_t, j_diff).mean()
            sim_trainer_A.ed_optimizers[idx].zero_grad()
            a_loss.backward()

            g_flat = get_flat_grad(sim_trainer_A.ed_primary_actors[idx])
            step_g_norms.append(g_flat.norm(2).item())
            sim_trainer_A.ed_optimizers[idx].step()

        cloud_st = states_t[:, -11:]
        cloud_probs = sim_trainer_A.cloud_primary_actor(cloud_st)
        j_diff_c = actions_t.clone()
        j_diff_c[:, -5:] = cloud_probs
        c_loss = -sim_trainer_A.critic(states_t, j_diff_c).mean()
        sim_trainer_A.cloud_optimizer.zero_grad()
        c_loss.backward()
        sim_trainer_A.cloud_optimizer.step()

        hybrid_grad_norms.append(np.mean(step_g_norms))

    with torch.no_grad():
        ed0_p_hyb_fin = sim_trainer_A.ed_primary_actors[0](eval_ed0_st).squeeze(0)
        cloud_p_hyb_fin = sim_trainer_A.cloud_primary_actor(eval_cloud_st).squeeze(0)

    ed0_ent_hyb_fin = compute_entropy(ed0_p_hyb_fin)
    ed0_max_hyb_fin = ed0_p_hyb_fin.max().item()
    cloud_ent_hyb_fin = compute_entropy(cloud_p_hyb_fin)
    cloud_max_hyb_fin = cloud_p_hyb_fin.max().item()

    sim_trainer_B = LRMATrainer(num_ed=25, num_mes=5, device=device_str)
    sim_trainer_B.critic.load_state_dict(trainer.critic.state_dict())
    for k in range(25):
        sim_trainer_B.ed_primary_actors[k].load_state_dict(trainer.ed_primary_actors[k].state_dict())
    sim_trainer_B.cloud_primary_actor.load_state_dict(trainer.cloud_primary_actor.state_dict())

    full_grad_norms = []
    for step in range(1, 21):
        curr_ed_probs = [sim_trainer_B.ed_primary_actors[k](states_t[:, k*9:(k+1)*9]) for k in range(25)]
        curr_cloud_probs = sim_trainer_B.cloud_primary_actor(states_t[:, -11:])
        a_full = torch.cat(curr_ed_probs + [curr_cloud_probs], dim=-1)

        loss_full = -sim_trainer_B.critic(states_t, a_full).mean()

        for k in range(25):
            sim_trainer_B.ed_optimizers[k].zero_grad()
        sim_trainer_B.cloud_optimizer.zero_grad()

        loss_full.backward()

        step_g_norms = []
        for k in range(25):
            g_flat = get_flat_grad(sim_trainer_B.ed_primary_actors[k])
            step_g_norms.append(g_flat.norm(2).item())
            sim_trainer_B.ed_optimizers[k].step()
        sim_trainer_B.cloud_optimizer.step()

        full_grad_norms.append(np.mean(step_g_norms))

    with torch.no_grad():
        ed0_p_full_fin = sim_trainer_B.ed_primary_actors[0](eval_ed0_st).squeeze(0)
        cloud_p_full_fin = sim_trainer_B.cloud_primary_actor(eval_cloud_st).squeeze(0)

    ed0_ent_full_fin = compute_entropy(ed0_p_full_fin)
    ed0_max_full_fin = ed0_p_full_fin.max().item()
    cloud_ent_full_fin = compute_entropy(cloud_p_full_fin)
    cloud_max_full_fin = cloud_p_full_fin.max().item()

    print("\n============================================================")
    print("PART 8: OBJECTIVE COMPARISON SUMMARY TABLE")
    print("============================================================")
    print(f"{'Metric':<30} | {'Hybrid Replay Objective':<25} | {'Full Current Policy Objective':<28}")
    print("-" * 88)
    hyb_str = f"{ed0_ent_start:.4f} -> {ed0_ent_hyb_fin:.4f}"
    full_str = f"{ed0_ent_start:.4f} -> {ed0_ent_full_fin:.4f}"
    print(f"{'ED Entropy (Start -> Final)':<30} | {hyb_str:<25} | {full_str:<28}")

    hyb_m_str = f"{ed0_max_start:.4f} -> {ed0_max_hyb_fin:.4f}"
    full_m_str = f"{ed0_max_start:.4f} -> {ed0_max_full_fin:.4f}"
    print(f"{'ED MaxP (Start -> Final)':<30} | {hyb_m_str:<25} | {full_m_str:<28}")

    cloud_str = f"{cloud_ent_start:.4f} -> {cloud_ent_hyb_fin:.4f}"
    cloud_f_str = f"{cloud_ent_start:.4f} -> {cloud_ent_full_fin:.4f}"
    print(f"{'Cloud Entropy (Start -> Final)':<30} | {cloud_str:<25} | {cloud_f_str:<28}")

    cloud_m_str = f"{cloud_max_start:.4f} -> {cloud_max_hyb_fin:.4f}"
    cloud_mf_str = f"{cloud_max_start:.4f} -> {cloud_max_full_fin:.4f}"
    print(f"{'Cloud MaxP (Start -> Final)':<30} | {cloud_m_str:<25} | {cloud_mf_str:<28}")

    print(f"{'Mean Actor Gradient Norm':<30} | {np.mean(hybrid_grad_norms):<25.6f} | {np.mean(full_grad_norms):<28.6f}")

    print("\n============================================================")
    print("PART 9: FINAL CLASSIFICATION")
    print("============================================================")

    both_similar = (abs(ed0_ent_hyb_fin - ed0_ent_full_fin) < 0.1 and avg_sim > 0.8)

    if both_similar:
        classification = "A. BOTH OBJECTIVES BEHAVE SIMILARLY"
    elif ed0_ent_hyb_fin < 0.1 and ed0_ent_full_fin > 0.4:
        classification = "B. HYBRID REPLAY OBJECTIVE CAUSES POLICY COLLAPSE"
    elif ed0_ent_full_fin < 0.1 and ed0_ent_hyb_fin > 0.4:
        classification = "C. FULL CURRENT-POLICY OBJECTIVE CAUSES POLICY COLLAPSE"
    elif avg_sim < 0.5:
        classification = "D. HYBRID OBJECTIVE PRODUCES A DIFFERENT GRADIENT DIRECTION"
    else:
        classification = "E. ACTOR OBJECTIVE IS NOT THE PRIMARY CAUSE"

    print(f"CLASSIFICATION: {classification}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("HISTORICAL RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
