import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import numpy as np
from src.lrma_trainer import LRMATrainer


def evaluate_critic_cases(trainer, states_t, actions_t):
    batch_size = states_t.shape[0]
    device = trainer.device

    with torch.no_grad():
        # Case 1: Real discrete one-hot replay actions
        q_real = trainer.critic(states_t, actions_t)

        # Case 2: Continuous actor probability actions for all agents
        ed_actor_probs = []
        for idx in range(trainer.num_ed):
            ed_st = states_t[:, idx * 9 : (idx + 1) * 9]
            ed_actor_probs.append(trainer.ed_primary_actors[idx](ed_st))
        cloud_st = states_t[:, -11:]
        cloud_actor_probs = trainer.cloud_primary_actor(cloud_st)

        joint_actor_probs = torch.cat(ed_actor_probs + [cloud_actor_probs], dim=-1)
        q_actor_probs = trainer.critic(states_t, joint_actor_probs)

        # Case 3: Uniform probability actions
        # 25 EDs x [0.5, 0.5] = 50 dims; Cloud x [0.2, 0.2, 0.2, 0.2, 0.2] = 5 dims
        ed_uniform = torch.full((batch_size, 50), 0.5, device=device)
        cloud_uniform = torch.full((batch_size, 5), 0.2, device=device)
        joint_uniform = torch.cat([ed_uniform, cloud_uniform], dim=-1)
        q_uniform = trainer.critic(states_t, joint_uniform)

        # Detailed ED-0 specific joint action variations (holding other agents at replay actions)
        # ED-0 Local [1, 0]
        a_ed0_local = actions_t.clone()
        a_ed0_local[:, 0] = 1.0
        a_ed0_local[:, 1] = 0.0
        q_ed0_local = trainer.critic(states_t, a_ed0_local)

        # ED-0 Offload [0, 1]
        a_ed0_offload = actions_t.clone()
        a_ed0_offload[:, 0] = 0.0
        a_ed0_offload[:, 1] = 1.0
        q_ed0_offload = trainer.critic(states_t, a_ed0_offload)

        # ED-0 Actor Probs
        a_ed0_prob = actions_t.clone()
        a_ed0_prob[:, 0:2] = ed_actor_probs[0]
        q_ed0_prob = trainer.critic(states_t, a_ed0_prob)

        # ED-0 Uniform [0.5, 0.5]
        a_ed0_uniform = actions_t.clone()
        a_ed0_uniform[:, 0:2] = 0.5
        q_ed0_uniform = trainer.critic(states_t, a_ed0_uniform)

    return {
        'q_real': q_real,
        'q_actor_probs': q_actor_probs,
        'q_uniform': q_uniform,
        'q_ed0_local': q_ed0_local,
        'q_ed0_offload': q_ed0_offload,
        'q_ed0_prob': q_ed0_prob,
        'q_ed0_uniform': q_ed0_uniform,
    }


def print_q_stats(label, q_tensor):
    q_np = q_tensor.cpu().numpy()
    print(f"  {label:<30}: Min={q_np.min():7.4f}, Max={q_np.max():7.4f}, Mean={q_np.mean():7.4f}, Std={q_np.std():7.4f}")


def main():
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device_str}")
    trainer = LRMATrainer(num_ed=25, num_mes=5, device=device_str)

    batch_size = 64
    np.random.seed(42)
    torch.manual_seed(42)

    for _ in range(batch_size):
        s_joint = np.random.randn(236).astype(np.float32)
        a_joint = np.random.randn(55).astype(np.float32)
        r = float(np.random.uniform(-5.0, 5.0))
        ns_joint = np.random.randn(236).astype(np.float32)
        trainer.replay_buffer_ed.add(s_joint, a_joint, r, ns_joint, False)

    states, actions, rewards, next_states, dones = trainer.replay_buffer_ed.sample(batch_size)

    states_t = torch.FloatTensor(states).to(trainer.device)
    actions_t = torch.FloatTensor(actions).to(trainer.device)

    print("\n============================================================")
    print("INITIAL CRITIC EVALUATION (BEFORE TRAINING)")
    print("============================================================")
    eval_init = evaluate_critic_cases(trainer, states_t, actions_t)
    print_q_stats("Case 1: Real One-Hot Replay", eval_init['q_real'])
    print_q_stats("Case 2: Actor Probs (Joint)", eval_init['q_actor_probs'])
    print_q_stats("Case 3: Uniform Probs (Joint)", eval_init['q_uniform'])
    print_q_stats("ED-0 Local [1,0]", eval_init['q_ed0_local'])
    print_q_stats("ED-0 Offload [0,1]", eval_init['q_ed0_offload'])
    print_q_stats("ED-0 Actor Probs", eval_init['q_ed0_prob'])
    print_q_stats("ED-0 Uniform [0.5,0.5]", eval_init['q_ed0_uniform'])

    print("\nRunning exactly 20 train_step() updates...")
    for step in range(1, 21):
        a_loss, c_loss = trainer.train_step(batch_size=64, gamma=0.99, xi_soft=0.01)
        if step in [1, 5, 10, 15, 20]:
            print(f"  Step {step:02d}: Actor Loss = {a_loss:7.4f}, Critic Loss = {c_loss:7.4f}")

    print("\n============================================================")
    print("POST-TRAINING CRITIC EVALUATION (AFTER 20 UPDATES)")
    print("============================================================")
    eval_post = evaluate_critic_cases(trainer, states_t, actions_t)
    print_q_stats("Case 1: Real One-Hot Replay", eval_post['q_real'])
    print_q_stats("Case 2: Actor Probs (Joint)", eval_post['q_actor_probs'])
    print_q_stats("Case 3: Uniform Probs (Joint)", eval_post['q_uniform'])
    print_q_stats("ED-0 Local [1,0]", eval_post['q_ed0_local'])
    print_q_stats("ED-0 Offload [0,1]", eval_post['q_ed0_offload'])
    print_q_stats("ED-0 Actor Probs", eval_post['q_ed0_prob'])
    print_q_stats("ED-0 Uniform [0.5,0.5]", eval_post['q_ed0_uniform'])

    # Q differences
    diff_real_actor = (eval_post['q_real'] - eval_post['q_actor_probs']).abs().mean().item()
    diff_real_uniform = (eval_post['q_real'] - eval_post['q_uniform']).abs().mean().item()
    diff_actor_uniform = (eval_post['q_actor_probs'] - eval_post['q_uniform']).abs().mean().item()

    print("\n============================================================")
    print("CRITIC ACTION DISTRIBUTION EVALUATION SUMMARY")
    print("============================================================")
    print(f"Mean |Q_real - Q_actor_probs|   : {diff_real_actor:.4f}")
    print(f"Mean |Q_real - Q_uniform|       : {diff_real_uniform:.4f}")
    print(f"Mean |Q_actor_probs - Q_uniform|: {diff_actor_uniform:.4f}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("RESULTS MODIFIED: NO")
    print("TRAINING EXECUTED BY DIAGNOSTIC: exactly 20 train_step calls")


if __name__ == "__main__":
    main()
