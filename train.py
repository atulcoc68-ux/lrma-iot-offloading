import os
import sys
import time
import argparse
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_lrma_agent(seed=42, num_ed=EnvConfig.NUM_ED, V_val=EnvConfig.V,
                     reset_enabled=True, task_arrival_rate=0.6, total_slots=EnvConfig.TOTAL_SLOTS):
    """
    Implements LRMA Algorithm Train Framework (Paper Algorithm 1, Section V).
    Centralized Training with Distributed Execution (CTDE) for N ED Actors + 1 Cloud Actor.
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

    # 3. Multi-Agent Setup (N ED Actors + 1 Cloud Actor)
    state_dim_ed = 9
    action_dim_ed = num_ed + 1
    
    state_dim_cloud = 5 + EnvConfig.NUM_MES + 1
    action_dim_cloud = EnvConfig.NUM_MES

    ed_primary_actors = [DRLActor(state_dim_ed, action_dim_ed) for _ in range(num_ed)]
    ed_target_actors = [DRLActor(state_dim_ed, action_dim_ed) for _ in range(num_ed)]
    
    cloud_primary_actor = DRLActor(state_dim_cloud, action_dim_cloud)
    cloud_target_actor = DRLActor(state_dim_cloud, action_dim_cloud)

    for p, t_net in zip(ed_primary_actors, ed_target_actors):
        t_net.load_state_dict(p.state_dict())
    cloud_target_actor.load_state_dict(cloud_primary_actor.state_dict())

    joint_state_dim = state_dim_ed * num_ed + state_dim_cloud
    joint_action_dim = action_dim_ed * num_ed + action_dim_cloud
    critic = DRLCritic(joint_state_dim, joint_action_dim)

    ed_optimizers = [torch.optim.Adam(actor.parameters(), lr=EnvConfig.LR_ACTOR) for actor in ed_primary_actors]
    cloud_optimizer = torch.optim.Adam(cloud_primary_actor.parameters(), lr=EnvConfig.LR_ACTOR)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=EnvConfig.LR_CRITIC)

    reward_calc = LRMARewardCalculator(V_penalty=V_val)
    env = LRMA_Environment(loader, predictor, EnvConfig, num_ed=num_ed, V_val=V_val)

    # Replay Buffers (Algorithm 1, line 2)
    replay_buffer_ed = []
    replay_buffer_cloud = []

    slot_rewards = []
    task_delays = []
    task_processing_delays = []
    task_waiting_times = []
    offload_decisions = []
    ed_queue_history = []
    mes_queue_history = []

    start_time = time.time()

    # Time-slot simulation loop t = 1 ... K (Algorithm 1, lines 4-32)
    for t in range(1, total_slots + 1):
        slot_workload_ed = workload_by_slot.get(t, {})
        env.update_time_slot(t, slot_workload_ed)

        # Train LSTM Predictor periodically on history (Algorithm 1, lines 6-7)
        if len(env.history_arrival_states) > EnvConfig.SEQ_LENGTH + 5 and t % 30 == 0:
            train_predictor(predictor, env.history_arrival_states, epochs=1)

        slot_reward_sum = 0.0

        # Process per-ED task arrivals
        for ed_idx in range(num_ed):
            ed_tasks = slot_workload_ed.get(ed_idx, [])
            ed_actor = ed_primary_actors[ed_idx]

            for task in ed_tasks:
                # Get local ED state for ED agent ed_idx (Paper Eq. 34)
                s_ed = env.get_ed_state(ed_idx, task, ed_tasks)
                candidates_ed, probs_ed = point_to_uniform_quantization(ed_actor, s_ed)
                ed_action = candidates_ed[0]

                # Cloud decision for offloaded tasks (Algorithm 1, lines 16-19)
                if ed_action > 0:
                    flat_offloaded = [t_item for e_id in range(num_ed) for t_item in slot_workload_ed.get(e_id, [])]
                    s_cloud = env.get_cloud_state(task, flat_offloaded)
                    candidates_cloud, _ = point_to_uniform_quantization(cloud_primary_actor, s_cloud)
                    cloud_action = candidates_cloud[0]
                else:
                    s_cloud = np.zeros(state_dim_cloud, dtype=np.float32)
                    cloud_action = 0

                # Environment step (Algorithm 1, line 30)
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

                # Store experience in buffer (Algorithm 1, line 22)
                replay_buffer_ed.append((s_ed, ed_action, r_tot))
                if res['is_offloaded']:
                    replay_buffer_cloud.append((s_cloud, cloud_action, r_tot))

        slot_rewards.append(slot_reward_sum)
        ed_queue_history.append(float(np.mean(env.q_device)))
        mes_queue_history.append(float(np.mean(env.q_es)))

        # Soft update target network & Parameter Reset (Algorithm 1, lines 23-29)
        if t % EnvConfig.UPDATE_INTERVAL == 0:
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

    total_gen = len(task_delays)
    completed_count = sum(1 for d in task_delays if d < total_slots)
    pending_count = total_gen - completed_count
    mean_comp_delay = float(np.mean(task_delays)) if task_delays else 0.0
    max_comp_delay = float(np.max(task_delays)) if task_delays else 0.0
    mean_proc_delay = float(np.mean(task_processing_delays)) if task_processing_delays else 0.0
    mean_wait_time = float(np.mean(task_waiting_times)) if task_waiting_times else 0.0
    mean_backlog = float(np.mean(ed_queue_history) + np.mean(mes_queue_history)) / 8e6

    print("\n================ Delay Accounting Audit Sanity Summary ================")
    print(f"Generated Tasks:       {total_gen}")
    print(f"Completed Tasks:       {completed_count}")
    print(f"Pending Tasks:         {pending_count}")
    print(f"Mean Completion Delay: {mean_comp_delay:.4f} s")
    print(f"Max Completion Delay:  {max_comp_delay:.4f} s")
    print(f"Mean Processing Time:  {mean_proc_delay:.4f} s")
    print(f"Mean Waiting Time:     {mean_wait_time:.4f} s")
    print(f"Mean Queue Backlog:    {mean_backlog:.4f} MB")
    print(f"Offloading Ratio:      {np.mean(offload_decisions):.4f}")
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
        'ed_queue_history': ed_queue_history,
        'mes_queue_history': mes_queue_history,
        'slot_rewards': slot_rewards
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
