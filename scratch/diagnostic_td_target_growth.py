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


def param_norm(model):
    norm_sq = 0.0
    for p in model.parameters():
        norm_sq += p.norm(2).item() ** 2
    return norm_sq ** 0.5


def model_distance(model1, model2):
    dist_sq = 0.0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        dist_sq += (p1 - p2).norm(2).item() ** 2
    return dist_sq ** 0.5


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
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(trainer.device)
    next_states_t = torch.FloatTensor(next_states).to(trainer.device)
    dones_t = torch.FloatTensor(dones).unsqueeze(1).to(trainer.device)

    # INITIAL DIAGNOSTIC
    with torch.no_grad():
        current_Q_init = trainer.critic(states_t, actions_t)
        
        target_ed_actions_init = []
        for idx in range(trainer.num_ed):
            ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
            target_ed_actions_init.append(trainer.ed_target_actors[idx](ed_st))
        cloud_st = next_states_t[:, -11:]
        cloud_target_probs_init = trainer.cloud_target_actor(cloud_st)
        target_joint_action_init = torch.cat(target_ed_actions_init + [cloud_target_probs_init], dim=-1)

        target_Q_init = trainer.critic_target(next_states_t, target_joint_action_init)
        gamma = 0.99
        target_y_init = rewards_t + (1.0 - dones_t) * gamma * target_Q_init
        td_error_init = target_y_init - current_Q_init

    s_cQ_init = get_stats(current_Q_init)
    s_tQ_init = get_stats(target_Q_init)
    s_ty_init = get_stats(target_y_init)
    s_tde_init = get_stats(td_error_init)
    s_rew = get_stats(rewards_t)

    print("============================================================")
    print("INITIAL TD DIAGNOSTIC (BEFORE TRAINING)")
    print("============================================================")
    print(f"Reward    : min={s_rew['min']:.4f}, max={s_rew['max']:.4f}, mean={s_rew['mean']:.4f}, std={s_rew['std']:.4f}")
    print(f"Current Q : min={s_cQ_init['min']:.4f}, max={s_cQ_init['max']:.4f}, mean={s_cQ_init['mean']:.4f}, std={s_cQ_init['std']:.4f}")
    print(f"Target Q  : min={s_tQ_init['min']:.4f}, max={s_tQ_init['max']:.4f}, mean={s_tQ_init['mean']:.4f}, std={s_tQ_init['std']:.4f}")
    print(f"Target Y  : min={s_ty_init['min']:.4f}, max={s_ty_init['max']:.4f}, mean={s_ty_init['mean']:.4f}, std={s_ty_init['std']:.4f}")
    print(f"TD Error  : min={s_tde_init['min']:.4f}, max={s_tde_init['max']:.4f}, mean={s_tde_init['mean']:.4f}, std={s_tde_init['std']:.4f}")

    print("\n============================================================")
    print("20-STEP TD TARGET GROWTH DIAGNOSTIC")
    print("============================================================")

    history = []

    for step in range(1, 21):
        # 1. Target Q & Target Y calculation
        with torch.no_grad():
            target_ed_actions = []
            for idx in range(trainer.num_ed):
                ed_st = next_states_t[:, idx * 9 : (idx + 1) * 9]
                target_ed_actions.append(trainer.ed_target_actors[idx](ed_st))
            cloud_st = next_states_t[:, -11:]
            cloud_target_probs = trainer.cloud_target_actor(cloud_st)
            target_joint_action = torch.cat(target_ed_actions + [cloud_target_probs], dim=-1)

            target_Q = trainer.critic_target(next_states_t, target_joint_action)
            target_y = rewards_t + (1.0 - dones_t) * gamma * target_Q

        # Current Q before update
        current_Q = trainer.critic(states_t, actions_t)
        td_error = target_y - current_Q
        critic_loss = nn.MSELoss()(current_Q, target_y)

        # Critic update with gradient norm measurement
        trainer.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        critic_grad_norm = 0.0
        for p in trainer.critic.parameters():
            if p.grad is not None:
                critic_grad_norm += p.grad.norm(2).item() ** 2
        critic_grad_norm = critic_grad_norm ** 0.5
        
        trainer.critic_optimizer.step()

        # Actor updates
        for idx in range(trainer.num_ed):
            trainer.ed_optimizers[idx].zero_grad()
            pred_probs = trainer.ed_primary_actors[idx](states_t[:, idx*9:(idx+1)*9])
            joint_a_diff = actions_t.clone()
            joint_a_diff[:, idx*2:(idx+1)*2] = pred_probs
            actor_loss = -trainer.critic(states_t, joint_a_diff).mean()
            actor_loss.backward()
            trainer.ed_optimizers[idx].step()

        trainer.cloud_optimizer.zero_grad()
        cloud_probs = trainer.cloud_primary_actor(states_t[:, -11:])
        joint_a_cloud_diff = actions_t.clone()
        joint_a_cloud_diff[:, -trainer.action_dim_cloud:] = cloud_probs
        cloud_loss = -trainer.critic(states_t, joint_a_cloud_diff).mean()
        cloud_loss.backward()
        trainer.cloud_optimizer.step()

        # Soft updates
        for p, trg in zip(trainer.ed_primary_actors, trainer.ed_target_actors):
            soft_update(trg, p, 0.01)
        soft_update(trainer.cloud_target_actor, trainer.cloud_primary_actor, 0.01)
        soft_update(trainer.critic_target, trainer.critic, 0.01)

        # Norms & Distances
        p_norm_critic = param_norm(trainer.critic)
        p_norm_critic_target = param_norm(trainer.critic_target)
        c_dist = model_distance(trainer.critic, trainer.critic_target)

        scQ = get_stats(current_Q)
        stQ = get_stats(target_Q)
        sty = get_stats(target_y)
        stde = get_stats(td_error)

        with torch.no_grad():
            ed0_p = trainer.ed_primary_actors[0](states_t[0:1, 0:9]).squeeze(0)
            cloud_p = trainer.cloud_primary_actor(states_t[0:1, -11:]).squeeze(0)

        history.append({
            'step': step,
            'cQ': scQ,
            'tQ': stQ,
            'ty': sty,
            'tde': stde,
            'critic_loss': critic_loss.item(),
            'grad_norm': critic_grad_norm,
            'c_norm': p_norm_critic,
            'ct_norm': p_norm_critic_target,
            'c_dist': c_dist,
            'ed0_p': ed0_p.tolist(),
            'cloud_p': cloud_p.tolist()
        })

        print(f"UPDATE {step:02d}")
        print(f"  Current Q: min={scQ['min']:7.4f}, max={scQ['max']:7.4f}, mean={scQ['mean']:7.4f}, std={scQ['std']:7.4f}")
        print(f"  Target Q : min={stQ['min']:7.4f}, max={stQ['max']:7.4f}, mean={stQ['mean']:7.4f}, std={stQ['std']:7.4f}")
        print(f"  Target Y : min={sty['min']:7.4f}, max={sty['max']:7.4f}, mean={sty['mean']:7.4f}, std={sty['std']:7.4f}")
        print(f"  TD Error : min={stde['min']:7.4f}, max={stde['max']:7.4f}, mean={stde['mean']:7.4f}, std={stde['std']:7.4f}")
        print(f"  Critic Loss: {critic_loss.item():.6f} | Critic Grad Norm: {critic_grad_norm:.6f}")
        print(f"  Critic Norm: {p_norm_critic:.4f} | Target Critic Norm: {p_norm_critic_target:.4f} | Critic-Target Dist: {c_dist:.6f}")
        print(f"  ED-0 Probs: {[round(x,4) for x in ed0_p.tolist()]} | Cloud Probs: {[round(x,4) for x in cloud_p.tolist()]}\n")

    # Final Decision Assessment
    init_cQ_mean = history[0]['cQ']['mean']
    final_cQ_mean = history[-1]['cQ']['mean']

    init_tQ_mean = history[0]['tQ']['mean']
    final_tQ_mean = history[-1]['tQ']['mean']

    init_ty_mean = history[0]['ty']['mean']
    final_ty_mean = history[-1]['ty']['mean']

    tQ_growth = (final_tQ_mean - init_tQ_mean) > 1.0
    ty_growth = (final_ty_mean - init_ty_mean) > 1.0
    cQ_growth = (final_cQ_mean - init_cQ_mean) > 1.0
    grad_explosion = any(h['grad_norm'] > 50.0 for h in history)
    target_lag = history[-1]['c_dist'] > 2.0

    print("============================================================")
    print("FINAL DECISION ANALYSIS")
    print("============================================================")
    print(f"Initial vs Final Current Q Mean: {init_cQ_mean:.4f} -> {final_cQ_mean:.4f}")
    print(f"Initial vs Final Target Q Mean : {init_tQ_mean:.4f} -> {final_tQ_mean:.4f}")
    print(f"Initial vs Final Target Y Mean : {init_ty_mean:.4f} -> {final_ty_mean:.4f}")
    print(f"Final Critic-Target Distance  : {history[-1]['c_dist']:.6f}")
    print(f"Max Critic Grad Norm Observed : {max(h['grad_norm'] for h in history):.6f}")

    if tQ_growth:
        decision = "A. TARGET-Q GROWTH"
    elif ty_growth:
        decision = "B. TD-TARGET GROWTH"
    elif cQ_growth:
        decision = "C. CRITIC OVER-ESTIMATION"
    elif target_lag:
        decision = "D. TARGET-NETWORK LAG"
    elif grad_explosion:
        decision = "E. CRITIC GRADIENT EXPLOSION"
    else:
        decision = "F. NO CLEAR Q-GROWTH MECHANISM"

    print(f"\nOBSERVED PATTERN DECISION: {decision}")

    print("\nSOURCE MODIFIED: NO")
    print("DATASET MODIFIED: NO")
    print("FROZEN TRACE MODIFIED: NO")
    print("PERSISTENT RESULTS MODIFIED: NO")


if __name__ == "__main__":
    main()
