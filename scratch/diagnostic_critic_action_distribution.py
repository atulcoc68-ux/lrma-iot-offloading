import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from src.lrma_trainer import LRMATrainer


def get_q_stats(q_tensor):
    q_np = q_tensor.detach().cpu().numpy()
    return {
        'min': float(q_np.min()),
        'max': float(q_np.max()),
        'mean': float(q_np.mean()),
        'std': float(q_np.std())
    }


def compute_action_gradient(critic, states_t, actions_t):
    a_grad = actions_t.clone().detach().requires_grad_(True)
    q_val = critic(states_t, a_grad)
    q_val.sum().backward()
    grad = a_grad.grad.detach()
    
    grad_norm = float(grad.norm(2).item())
    grad_np = grad.cpu().numpy()
    return {
        'norm': grad_norm,
        'min': float(grad_np.min()),
        'max': float(grad_np.max()),
        'mean': float(grad_np.mean())
    }


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    # Populate replay buffer with 64 real-shaped CTDE transitions
    batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)

    for _ in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        
        # Real one-hot actions: 25 EDs (2-dim one-hot) + Cloud (5-dim one-hot)
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

    # TEST 1 — REAL REPLAY ACTIONS
    with torch.no_grad():
        q_real = trainer.critic(states_t, actions_t)
    stats_real = get_q_stats(q_real)

    # TEST 2 — CURRENT ACTOR PROBABILITY ACTIONS
    with torch.no_grad():
        ed_actor_probs = []
        for idx in range(trainer.num_ed):
            ed_st = states_t[:, idx * 9 : (idx + 1) * 9]
            ed_actor_probs.append(trainer.ed_primary_actors[idx](ed_st))
        cloud_st = states_t[:, -11:]
        cloud_actor_probs = trainer.cloud_primary_actor(cloud_st)

        actor_probability_joint = torch.cat(ed_actor_probs + [cloud_actor_probs], dim=-1)
        q_actor = trainer.critic(states_t, actor_probability_joint)
    stats_actor = get_q_stats(q_actor)

    # TEST 3 — UNIFORM PROBABILITY ACTIONS
    with torch.no_grad():
        ed_uniform = torch.full((batch_size, 50), 0.5, device=trainer.device)
        cloud_uniform = torch.full((batch_size, 5), 0.2, device=trainer.device)
        uniform_joint_action = torch.cat([ed_uniform, cloud_uniform], dim=-1)
        q_uniform = trainer.critic(states_t, uniform_joint_action)
    stats_uniform = get_q_stats(q_uniform)

    # TEST 4 — ACTION REPRESENTATION COMPARISON
    diff_actor_replay = {k: stats_actor[k] - stats_real[k] for k in ['mean', 'std', 'min', 'max']}
    diff_uniform_replay = {k: stats_uniform[k] - stats_real[k] for k in ['mean', 'std', 'min', 'max']}
    diff_actor_uniform = {k: stats_actor[k] - stats_uniform[k] for k in ['mean', 'std', 'min', 'max']}

    # TEST 5 — Q GRADIENT WITH RESPECT TO ACTION (dQ/dA)
    grad_real = compute_action_gradient(trainer.critic, states_t, actions_t)
    grad_actor = compute_action_gradient(trainer.critic, states_t, actor_probability_joint)
    grad_uniform = compute_action_gradient(trainer.critic, states_t, uniform_joint_action)

    # TEST 6 — ACTION VALIDITY
    # Replay one-hot validity check
    replay_ed_one_hot = all(
        torch.allclose(actions_t[:, i*2:(i+1)*2].sum(dim=-1), torch.ones(batch_size, device=trainer.device))
        for i in range(25)
    )
    replay_cloud_one_hot = torch.allclose(actions_t[:, -5:].sum(dim=-1), torch.ones(batch_size, device=trainer.device))

    # Actor probability sum check
    actor_sums_valid = (
        all(torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, device=trainer.device)) for probs in ed_actor_probs) and
        torch.allclose(cloud_actor_probs.sum(dim=-1), torch.ones(batch_size, device=trainer.device))
    )

    # Uniform probability sum check
    uniform_sums_valid = (
        torch.allclose(ed_uniform.view(batch_size, 25, 2).sum(dim=-1), torch.ones(batch_size, 25, device=trainer.device)) and
        torch.allclose(cloud_uniform.sum(dim=-1), torch.ones(batch_size, device=trainer.device))
    )

    # Finite check
    all_finite = (
        torch.isfinite(actions_t).all().item() and
        torch.isfinite(actor_probability_joint).all().item() and
        torch.isfinite(uniform_joint_action).all().item() and
        torch.isfinite(q_real).all().item() and
        torch.isfinite(q_actor).all().item() and
        torch.isfinite(q_uniform).all().item()
    )

    print("============================================================")
    print("CRITIC ACTION REPRESENTATION DIAGNOSTIC")
    print("============================================================")

    print(f"Replay action:")
    print(f"    shape = {tuple(actions_t.shape)}")
    print(f"    Q min/max/mean/std = {stats_real['min']:.4f} / {stats_real['max']:.4f} / {stats_real['mean']:.4f} / {stats_real['std']:.4f}")

    print(f"\nActor probability action:")
    print(f"    shape = {tuple(actor_probability_joint.shape)}")
    print(f"    Q min/max/mean/std = {stats_actor['min']:.4f} / {stats_actor['max']:.4f} / {stats_actor['mean']:.4f} / {stats_actor['std']:.4f}")

    print(f"\nUniform probability action:")
    print(f"    shape = {tuple(uniform_joint_action.shape)}")
    print(f"    Q min/max/mean/std = {stats_uniform['min']:.4f} / {stats_uniform['max']:.4f} / {stats_uniform['mean']:.4f} / {stats_uniform['std']:.4f}")

    print(f"\nQ DIFFERENCES:")
    print(f"    Actor - Replay   = mean: {diff_actor_replay['mean']:+.4f}, std: {diff_actor_replay['std']:+.4f}, min: {diff_actor_replay['min']:+.4f}, max: {diff_actor_replay['max']:+.4f}")
    print(f"    Uniform - Replay = mean: {diff_uniform_replay['mean']:+.4f}, std: {diff_uniform_replay['std']:+.4f}, min: {diff_uniform_replay['min']:+.4f}, max: {diff_uniform_replay['max']:+.4f}")
    print(f"    Actor - Uniform  = mean: {diff_actor_uniform['mean']:+.4f}, std: {diff_actor_uniform['std']:+.4f}, min: {diff_actor_uniform['min']:+.4f}, max: {diff_actor_uniform['max']:+.4f}")

    print(f"\nACTION GRADIENTS (dQ/dA):")
    print(f"    Replay  = norm: {grad_real['norm']:.4f}, min: {grad_real['min']:.4f}, max: {grad_real['max']:.4f}, mean: {grad_real['mean']:.4f}")
    print(f"    Actor   = norm: {grad_actor['norm']:.4f}, min: {grad_actor['min']:.4f}, max: {grad_actor['max']:.4f}, mean: {grad_actor['mean']:.4f}")
    print(f"    Uniform = norm: {grad_uniform['norm']:.4f}, min: {grad_uniform['min']:.4f}, max: {grad_uniform['max']:.4f}, mean: {grad_uniform['mean']:.4f}")

    print(f"\nACTION VALIDITY:")
    print(f"    Replay one-hot: {'PASS' if (replay_ed_one_hot and replay_cloud_one_hot) else 'FAIL'}")
    print(f"    Actor probs sum=1: {'PASS' if actor_sums_valid else 'FAIL'}")
    print(f"    Uniform probs sum=1: {'PASS' if uniform_sums_valid else 'FAIL'}")
    print(f"    Finite tensors: {'PASS' if all_finite else 'FAIL'}")

    substantial_mismatch = (
        abs(diff_actor_replay['mean']) > 0.5 or
        abs(diff_uniform_replay['mean']) > 0.5 or
        abs(diff_actor_replay['max']) > 1.0
    )

    print("\nFINAL DECISION:")
    if substantial_mismatch:
        print("POSSIBLE ACTION-REPRESENTATION MISMATCH")
    else:
        print("NO STRONG ACTION-REPRESENTATION MISMATCH OBSERVED")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("TRAINING EXECUTED: NO")


if __name__ == "__main__":
    main()
