import os
import sys
import time
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.config import EnvConfig
    from src.data_loader import AlibabaWorkloadLoader, LRMATask
    from src.lstm_model import WorkloadPredictor, train_predictor
    from src.agents import DRLActor, DRLCritic, soft_update, point_to_uniform_quantization
    from src.environment import LRMA_Environment
    from src.lyapunov import LRMARewardCalculator
except ModuleNotFoundError:
    from config import EnvConfig
    from data_loader import AlibabaWorkloadLoader, LRMATask
    from lstm_model import WorkloadPredictor, train_predictor
    from agents import DRLActor, DRLCritic, soft_update, point_to_uniform_quantization
    from environment import LRMA_Environment
    from lyapunov import LRMARewardCalculator


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_param_norm(model):
    """Computes L2 norm of model parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.requires_grad:
            total_norm_sq += float(p.data.norm(2).item() ** 2)
    return total_norm_sq ** 0.5


def compute_grad_norm(model):
    """Computes L2 norm of model gradients."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += float(p.grad.data.norm(2).item() ** 2)
    return total_norm_sq ** 0.5


def one_hot(index, size):
    """Helper to convert discrete integer index to one-hot numpy array."""
    vec = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vec[index] = 1.0
    return vec


def train_lrma_agent(seed=42, num_ed=EnvConfig.NUM_ED, V_val=EnvConfig.V,
                     reset_enabled=True, task_arrival_rate=0.6, total_slots=EnvConfig.TOTAL_SLOTS):
    """
    Implements LRMA Algorithm Train Framework (Paper Algorithm 1, Section V).
    Centralized Training with Distributed Execution (CTDE) for N ED Actors + 1 Cloud Actor.
    Refactored to paper-faithful binary ED action space (action_dim_ed = 2: 0=local, 1=offload)
    and 5-class Cloud MES action space (action_dim_cloud = 5: MES 0..4).
    """
    set_seed(seed)
    print(f"\n--- Initializing LRMA Multi-Agent Training (Seed={seed}, N={num_ed}, V={V_val}, Reset={reset_enabled}) ---")

    # 1. Load Dataset & Generate Workload Trace once (70% Train Split)
    loader = AlibabaWorkloadLoader()
    workload_by_slot = loader.generate_reproducible_slot_workload(
        dataset_split='train', seed=seed, num_ed=num_ed, total_slots=total_slots, arrival_rate=task_arrival_rate
    )

    # 2. Setup LSTM Workload Predictor
    predictor = WorkloadPredictor(input_dim=4, hidden_dim=64, output_dim=4, num_layers=2)

    # 3. Multi-Agent Setup (N ED Actors + 1 Cloud Actor + Centralized Critic)
    state_dim_ed = 9
    action_dim_ed = 2  # Binary ED decision: 0 = local execution, 1 = offloading intent
    
    state_dim_cloud = 5 + EnvConfig.NUM_MES + 1
    action_dim_cloud = EnvConfig.NUM_MES  # 5 MES server choices

    ed_primary_actors = [DRLActor(state_dim_ed, action_dim_ed) for _ in range(num_ed)]
    ed_target_actors = [DRLActor(state_dim_ed, action_dim_ed) for _ in range(num_ed)]
    
    cloud_primary_actor = DRLActor(state_dim_cloud, action_dim_cloud)
    cloud_target_actor = DRLActor(state_dim_cloud, action_dim_cloud)

    for p, t_net in zip(ed_primary_actors, ed_target_actors):
        t_net.load_state_dict(p.state_dict())
    cloud_target_actor.load_state_dict(cloud_primary_actor.state_dict())

    joint_state_dim = state_dim_ed * num_ed + state_dim_cloud
    joint_action_dim = action_dim_ed * num_ed + action_dim_cloud  # N*2 + 5
    critic = DRLCritic(joint_state_dim, joint_action_dim)
    critic_target = DRLCritic(joint_state_dim, joint_action_dim)
    critic_target.load_state_dict(critic.state_dict())

    ed_optimizers = [torch.optim.Adam(actor.parameters(), lr=EnvConfig.LR_ACTOR) for actor in ed_primary_actors]
    cloud_optimizer = torch.optim.Adam(cloud_primary_actor.parameters(), lr=EnvConfig.LR_ACTOR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=EnvConfig.LR_CRITIC)

    reward_calc = LRMARewardCalculator(V_penalty=V_val)
    env = LRMA_Environment(loader, predictor, EnvConfig, num_ed=num_ed, V_val=V_val)

    # Experience Replay Buffer for Centralized Training
    replay_buffer = []

    # Diagnostic Counters and Tracker Variables
    critic_update_count = 0
    actor_update_count = 0
    backward_count = 0
    optimizer_step_count = 0

    ed_local_count = 0
    ed_offload_count = 0
    cloud_mes_action_counts = np.zeros(EnvConfig.NUM_MES, dtype=np.int64)

    init_ed_actor_norm = compute_param_norm(ed_primary_actors[0])
    init_cloud_actor_norm = compute_param_norm(cloud_primary_actor)
    init_critic_norm = compute_param_norm(critic)

    last_critic_loss = 0.0
    last_actor_loss = 0.0
    last_critic_grad_norm = 0.0
    last_actor_grad_norm = 0.0

    dummy_task = LRMATask(0, 0, 0, 0, 0, 1.0)

    slot_rewards = []
    task_delays = []
    task_processing_delays = []
    task_waiting_times = []
    offload_decisions = []
    ed_queue_history = []
    mes_queue_history = []

    start_time = time.time()
    gamma = 0.95

    # Time-slot simulation loop t = 1 ... K (Algorithm 1, lines 4-32)
    for t in range(1, total_slots + 1):
        slot_workload_ed = workload_by_slot.get(t, {})
        env.update_time_slot(t, slot_workload_ed)

        # Train LSTM Predictor periodically on history (Algorithm 1, lines 6-7)
        if len(env.history_arrival_states) > EnvConfig.SEQ_LENGTH + 5 and t % 30 == 0:
            train_predictor(predictor, env.history_arrival_states, epochs=1)

        slot_reward_sum = 0.0

        # Build list of all flat tasks in current slot
        flat_offloaded = [t_item for e_id in range(num_ed) for t_item in slot_workload_ed.get(e_id, [])]

        # Process per-ED task arrivals
        for ed_idx in range(num_ed):
            ed_tasks = slot_workload_ed.get(ed_idx, [])
            ed_actor = ed_primary_actors[ed_idx]

            for task in ed_tasks:
                # 1. State observation before step
                s_ed_list = []
                for i in range(num_ed):
                    i_tasks = slot_workload_ed.get(i, [])
                    cur_t = task if i == ed_idx else (i_tasks[0] if i_tasks else dummy_task)
                    s_ed_list.append(env.get_ed_state(i, cur_t, i_tasks))
                s_ed_arr = np.array(s_ed_list, dtype=np.float32)  # (num_ed, state_dim_ed)

                s_ed = s_ed_list[ed_idx]
                candidates_ed, _ = point_to_uniform_quantization(ed_actor, s_ed)
                ed_action = candidates_ed[0]  # 0 = local, 1 = offload

                if ed_action == 0:
                    ed_local_count += 1
                else:
                    ed_offload_count += 1

                # Cloud decision for offloaded tasks (Algorithm 1, lines 16-19)
                if ed_action == 1:
                    s_cloud = env.get_cloud_state(task, flat_offloaded)
                    candidates_cloud, _ = point_to_uniform_quantization(cloud_primary_actor, s_cloud)
                    cloud_action = candidates_cloud[0]  # 0 .. 4
                    if 0 <= cloud_action < EnvConfig.NUM_MES:
                        cloud_mes_action_counts[cloud_action] += 1
                else:
                    s_cloud = np.zeros(state_dim_cloud, dtype=np.float32)
                    cloud_action = 0

                # Construct joint state and joint action vectors
                a_ed_onehots = [one_hot(ed_action if i == ed_idx else 0, action_dim_ed) for i in range(num_ed)]
                a_cloud_onehot = one_hot(cloud_action if ed_action == 1 else 0, action_dim_cloud)

                joint_state = np.concatenate([s_ed_arr.flatten(), s_cloud])
                joint_action = np.concatenate([np.array(a_ed_onehots).flatten(), a_cloud_onehot])

                # 2. Environment step (Algorithm 1, line 30)
                res = env.step_task_offloading(ed_idx, task, ed_action, cloud_action)
                
                # Compute Rewards (Paper Eq. 38, 39, 40)
                r_ed = reward_calc.calculate_ed_individual_reward(task.size, res['delay'], env.q_device[ed_idx], task.size, res['is_offloaded'])
                r_cloud = reward_calc.calculate_cloud_individual_reward([(task.size, res['delay'], res['mes_assigned'])], env.q_es, [task.size]*env.num_mes)
                r_all = r_ed + r_cloud
                r_tot = reward_calc.calculate_comprehensive_reward(r_ed, r_all)

                slot_reward_sum += r_tot
                task_delays.append(res['delay'])
                if 'processing_delay' in res:
                    task_processing_delays.append(res['processing_delay'])
                if 'waiting_time' in res:
                    task_waiting_times.append(res['waiting_time'])
                offload_decisions.append(1 if res['is_offloaded'] else 0)

                # 3. Next state observation after step
                next_s_ed_list = []
                for i in range(num_ed):
                    i_tasks = slot_workload_ed.get(i, [])
                    cur_t = task if i == ed_idx else (i_tasks[0] if i_tasks else dummy_task)
                    next_s_ed_list.append(env.get_ed_state(i, cur_t, i_tasks))
                next_s_ed_arr = np.array(next_s_ed_list, dtype=np.float32)
                next_s_cloud = env.get_cloud_state(task, flat_offloaded) if ed_action == 1 else np.zeros(state_dim_cloud, dtype=np.float32)

                next_joint_state = np.concatenate([next_s_ed_arr.flatten(), next_s_cloud])
                done = 1.0 if t == total_slots else 0.0

                # Store transition in replay buffer (Algorithm 1, line 22)
                replay_buffer.append((
                    joint_state, joint_action, float(r_tot), next_joint_state, done,
                    s_ed_arr, s_cloud, next_s_ed_arr, next_s_cloud
                ))

                # Maintain replay buffer capacity
                if len(replay_buffer) > 10000:
                    replay_buffer.pop(0)

                # -------------------------------------------------------------
                # 4. Centralized Critic & Discrete Actor Training Update
                # -------------------------------------------------------------
                if len(replay_buffer) >= EnvConfig.BATCH_SIZE:
                    batch = random.sample(replay_buffer, EnvConfig.BATCH_SIZE)

                    b_joint_state = torch.FloatTensor(np.array([item[0] for item in batch]))
                    b_joint_action = torch.FloatTensor(np.array([item[1] for item in batch]))
                    b_reward = torch.FloatTensor(np.array([item[2] for item in batch])).unsqueeze(1)
                    b_next_joint_state = torch.FloatTensor(np.array([item[3] for item in batch]))
                    b_done = torch.FloatTensor(np.array([item[4] for item in batch])).unsqueeze(1)

                    b_s_ed_arr = torch.FloatTensor(np.array([item[5] for item in batch]))    # (B, N, state_dim_ed)
                    b_s_cloud = torch.FloatTensor(np.array([item[6] for item in batch]))     # (B, state_dim_cloud)
                    b_next_s_ed_arr = torch.FloatTensor(np.array([item[7] for item in batch]))
                    b_next_s_cloud = torch.FloatTensor(np.array([item[8] for item in batch]))

                    # ---------------------------------------------------------
                    # A. Centralized Critic Update (TD Target)
                    # ---------------------------------------------------------
                    with torch.no_grad():
                        next_act_probs_ed = [ed_target_actors[i](b_next_s_ed_arr[:, i, :]) for i in range(num_ed)]
                        next_act_probs_cloud = cloud_target_actor(b_next_s_cloud)
                        next_joint_act_probs = torch.cat(next_act_probs_ed + [next_act_probs_cloud], dim=-1)

                        q_target_next = critic_target(b_next_joint_state, next_joint_act_probs)
                        y_target = b_reward + gamma * (1.0 - b_done) * q_target_next

                    q_current = critic(b_joint_state, b_joint_action)
                    critic_loss = nn.MSELoss()(q_current, y_target)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    last_critic_grad_norm = compute_grad_norm(critic)
                    critic_optimizer.step()

                    critic_update_count += 1
                    backward_count += 1
                    optimizer_step_count += 1
                    last_critic_loss = float(critic_loss.item())

                    # ---------------------------------------------------------
                    # B. Discrete Softmax Policy Gradient Actor Updates
                    # ---------------------------------------------------------
                    # ED Actors optimization
                    for i in range(num_ed):
                        probs_i = ed_primary_actors[i](b_s_ed_arr[:, i, :])  # (B, 2)

                        joint_probs_list = []
                        for j in range(num_ed):
                            if j == i:
                                joint_probs_list.append(probs_i)
                            else:
                                with torch.no_grad():
                                    joint_probs_list.append(ed_primary_actors[j](b_s_ed_arr[:, j, :]))
                        with torch.no_grad():
                            joint_probs_list.append(cloud_primary_actor(b_s_cloud))

                        joint_act_i = torch.cat(joint_probs_list, dim=-1)
                        q_eval_i = critic(b_joint_state, joint_act_i)
                        actor_loss_i = -q_eval_i.mean()

                        ed_optimizers[i].zero_grad()
                        actor_loss_i.backward()
                        last_actor_grad_norm = compute_grad_norm(ed_primary_actors[i])
                        ed_optimizers[i].step()

                        actor_update_count += 1
                        backward_count += 1
                        optimizer_step_count += 1
                        last_actor_loss = float(actor_loss_i.item())

                    # Cloud Actor optimization
                    probs_cloud = cloud_primary_actor(b_s_cloud)  # (B, 5)
                    joint_probs_list = []
                    for j in range(num_ed):
                        with torch.no_grad():
                            joint_probs_list.append(ed_primary_actors[j](b_s_ed_arr[:, j, :]))
                    joint_probs_list.append(probs_cloud)

                    joint_act_cloud = torch.cat(joint_probs_list, dim=-1)
                    q_eval_cloud = critic(b_joint_state, joint_act_cloud)
                    cloud_actor_loss = -q_eval_cloud.mean()

                    cloud_optimizer.zero_grad()
                    cloud_actor_loss.backward()
                    cloud_optimizer.step()

                    actor_update_count += 1
                    backward_count += 1
                    optimizer_step_count += 1

        slot_rewards.append(slot_reward_sum)
        ed_queue_history.append(float(np.mean(env.q_device)))
        mes_queue_history.append(float(np.mean(env.q_es)))

        # Soft update target networks & Parameter Reset (Algorithm 1, lines 23-29)
        if t % EnvConfig.UPDATE_INTERVAL == 0:
            soft_update(critic_target, critic, EnvConfig.XI_SOFT)
            for p, trg in zip(ed_primary_actors, ed_target_actors):
                soft_update(trg, p, EnvConfig.XI_SOFT)
            soft_update(cloud_target_actor, cloud_primary_actor, EnvConfig.XI_SOFT)

            # Parameter Reset mechanism (Algorithm 1, line 26-28)
            if reset_enabled and (t % EnvConfig.DELTA_RESET == 0):
                for p in ed_primary_actors:
                    p.reset_last_layer()
                cloud_primary_actor.reset_last_layer()

        if t % 50 == 0 or t == total_slots:
            avg_d = np.mean(task_delays[-50:]) if task_delays else 0.0
            print(f"Slot {t}/{total_slots} | Slot Reward: {slot_reward_sum:.2f} | Avg Delay (last 50): {avg_d:.4f} s | Offload Ratio: {np.mean(offload_decisions[-50:]):.2f}")

    elapsed = time.time() - start_time
    print(f"LRMA Training Complete in {elapsed:.2f} seconds.")

    final_ed_actor_norm = compute_param_norm(ed_primary_actors[0])
    final_cloud_actor_norm = compute_param_norm(cloud_primary_actor)
    final_critic_norm = compute_param_norm(critic)

    total_gen = len(task_delays)
    completed_count = sum(1 for d in task_delays if d < total_slots)
    pending_count = total_gen - completed_count
    mean_comp_delay = float(np.mean(task_delays)) if task_delays else 0.0
    max_comp_delay = float(np.max(task_delays)) if task_delays else 0.0
    mean_proc_delay = float(np.mean(task_processing_delays)) if task_processing_delays else 0.0
    mean_wait_time = float(np.mean(task_waiting_times)) if task_waiting_times else 0.0
    mean_backlog = float(np.mean(ed_queue_history) + np.mean(mes_queue_history)) / 8e6

    total_decisions = ed_local_count + ed_offload_count
    ed_local_ratio = float(ed_local_count / total_decisions) if total_decisions > 0 else 0.0
    ed_offload_ratio = float(ed_offload_count / total_decisions) if total_decisions > 0 else 0.0

    print("\n================ Delay Accounting Audit Sanity Summary ================")
    print(f"Generated Tasks:       {total_gen}")
    print(f"Completed Tasks:       {completed_count}")
    print(f"Pending Tasks:         {pending_count}")
    print(f"Mean Completion Delay: {mean_comp_delay:.4f} s")
    print(f"Max Completion Delay:  {max_comp_delay:.4f} s")
    print(f"Mean Processing Time:  {mean_proc_delay:.4f} s")
    print(f"Mean Waiting Time:     {mean_wait_time:.4f} s")
    print(f"Mean Queue Backlog:    {mean_backlog:.4f} MB")
    print(f"ED Local Decisions:    {ed_local_count} ({ed_local_ratio*100:.2f}%)")
    print(f"ED Offload Decisions:  {ed_offload_count} ({ed_offload_ratio*100:.2f}%)")
    print(f"Cloud MES Assignments: {cloud_mes_action_counts.tolist()}")
    print(f"Critic Step Count:     {critic_update_count}")
    print(f"Actor Step Count:      {actor_update_count}")
    print(f"Backward Count:        {backward_count}")
    print(f"Optimizer Step Count:  {optimizer_step_count}")
    print(f"ED Actor Norm Shift:   {init_ed_actor_norm:.4f} -> {final_ed_actor_norm:.4f}")
    print(f"Critic Norm Shift:     {init_critic_norm:.4f} -> {final_critic_norm:.4f}")
    print(f"Last Critic Loss:      {last_critic_loss:.6e}")
    print(f"Last Actor Loss:       {last_actor_loss:.6e}")
    print(f"Last Critic Grad Norm: {last_critic_grad_norm:.6f}")
    print(f"Last Actor Grad Norm:  {last_actor_grad_norm:.6f}")
    print("=======================================================================\n")

    # Save Checkpoint Models
    chkpt_path = os.path.join(EnvConfig.CHECKPOINTS_DIR, f"lrma_actor_N{num_ed}_V{int(V_val)}_reset{reset_enabled}.pth")
    torch.save(ed_primary_actors[0].state_dict(), chkpt_path)

    results_dict = {
        'seed': seed,
        'num_ed': num_ed,
        'V': V_val,
        'reset_enabled': reset_enabled,
        'task_arrival_rate': task_arrival_rate,
        'generated_tasks': total_gen,
        'completed_tasks': completed_count,
        'pending_tasks': pending_count,
        'all_task_delay': float(np.sum(task_delays)),
        'avg_task_delay': mean_comp_delay,
        'max_task_delay': max_comp_delay,
        'avg_processing_time': mean_proc_delay,
        'avg_waiting_time': mean_wait_time,
        'mean_queue_backlog_mb': mean_backlog,
        'offloading_ratio': float(np.mean(offload_decisions)),
        'ed_local_count': ed_local_count,
        'ed_offload_count': ed_offload_count,
        'ed_local_ratio': ed_local_ratio,
        'ed_offload_ratio': ed_offload_ratio,
        'cloud_mes_action_counts': cloud_mes_action_counts.tolist(),
        'ed_queue_history': ed_queue_history,
        'mes_queue_history': mes_queue_history,
        'slot_rewards': slot_rewards,
        'critic_update_count': critic_update_count,
        'actor_update_count': actor_update_count,
        'backward_count': backward_count,
        'optimizer_step_count': optimizer_step_count,
        'init_ed_actor_norm': init_ed_actor_norm,
        'final_ed_actor_norm': final_ed_actor_norm,
        'init_critic_norm': init_critic_norm,
        'final_critic_norm': final_critic_norm,
        'last_critic_loss': last_critic_loss,
        'last_actor_loss': last_actor_loss,
        'last_critic_grad_norm': last_critic_grad_norm,
        'last_actor_grad_norm': last_actor_grad_norm,
        'checkpoint_path': chkpt_path
    }

    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LRMA Multi-Agent Reinforcement Learning Training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-ed", type=int, default=EnvConfig.NUM_ED, help="Number of EDs N")
    parser.add_argument("--V", type=float, default=EnvConfig.V, help="Lyapunov V parameter")
    parser.add_argument("--no-reset", action="store_true", help="Disable parameter reset")
    parser.add_argument("--arrival-rate", type=float, default=0.6, help="Task arrival rate")
    parser.add_argument("--slots", type=int, default=EnvConfig.TOTAL_SLOTS, help="Total time slots")
    args = parser.parse_args()

    results = train_lrma_agent(
        seed=args.seed,
        num_ed=args.num_ed,
        V_val=args.V,
        reset_enabled=not args.no_reset,
        task_arrival_rate=args.arrival_rate,
        total_slots=args.slots
    )
