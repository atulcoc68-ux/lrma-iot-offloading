import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

try:
    from src.config import EnvConfig
    from src.data_loader import AlibabaWorkloadLoader, LRMATask
    from src.lstm_model import WorkloadPredictor
    from src.agents import DRLActor, MA3MCOActor, LMADDPGActor, DVCCOAgent
    from src.environment import LRMA_Environment
    from src.lyapunov import LRMARewardCalculator
except ModuleNotFoundError:
    from config import EnvConfig
    from data_loader import AlibabaWorkloadLoader, LRMATask
    from lstm_model import WorkloadPredictor
    from agents import DRLActor, MA3MCOActor, LMADDPGActor, DVCCOAgent
    from environment import LRMA_Environment
    from lyapunov import LRMARewardCalculator


def evaluate_policy(algorithm='LRMA', num_ed=EnvConfig.NUM_ED, V_val=EnvConfig.V,
                    queue_type='MHFQ', reset_enabled=True, task_arrival_rate=0.6,
                    seed=42, total_slots=EnvConfig.TOTAL_SLOTS):
    """
    Evaluates an offloading algorithm on real Alibaba workload trace (30% Evaluation Split).
    Guarantees all comparative algorithms receive the IDENTICAL slot workload sequence per seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    loader = AlibabaWorkloadLoader()
    
    # Generate/load reproducible slot workload from 30% Test split
    workload_by_slot = loader.generate_reproducible_slot_workload(
        dataset_split='test', seed=seed, num_ed=num_ed, total_slots=total_slots, arrival_rate=task_arrival_rate
    )

    predictor = WorkloadPredictor(input_dim=4, hidden_dim=64, output_dim=4, num_layers=2)
    predictor.eval()

    env = LRMA_Environment(loader, predictor, EnvConfig, queue_type=queue_type, num_ed=num_ed, V_val=V_val)
    
    state_dim_ed = 9
    action_dim_ed = 2
    state_dim_cloud = 5 + EnvConfig.NUM_MES + 1
    action_dim_cloud = EnvConfig.NUM_MES

    if algorithm == 'MA3MCO':
        actor_ed = MA3MCOActor(state_dim_ed, num_ed + 1)
        actor_cloud = MA3MCOActor(state_dim_cloud, action_dim_cloud)
    elif algorithm == 'L-MADDPG':
        actor_ed = LMADDPGActor(state_dim_ed, num_ed + 1)
        actor_cloud = LMADDPGActor(state_dim_cloud, action_dim_cloud)
    elif algorithm == 'DVCCO':
        actor_ed = DVCCOAgent(state_dim_ed, num_ed + 1)
        actor_cloud = DVCCOAgent(state_dim_cloud, action_dim_cloud)
    else:
        # LRMA Policy
        actor_ed = DRLActor(state_dim_ed, action_dim_ed)
        actor_cloud = DRLActor(state_dim_cloud, action_dim_cloud)
        chkpt_path = os.path.join(EnvConfig.CHECKPOINTS_DIR, f"lrma_actor_N{num_ed}_V{int(V_val)}_reset{reset_enabled}.pth")
        if os.path.exists(chkpt_path):
            actor_ed.load_state_dict(torch.load(chkpt_path))

    actor_ed.eval()
    actor_cloud.eval()

    task_delays = []
    task_energies = []
    offload_decisions = []
    ed_queue_history = []
    mes_queue_history = []
    task_records = []

    for t in range(1, total_slots + 1):
        # Support both string keys (if loaded from JSON) and int keys
        slot_workload_ed = workload_by_slot.get(t, workload_by_slot.get(str(t), {}))
        
        # Build flat list of all tasks across EDs for Cloud agent observation & global state
        if isinstance(slot_workload_ed, dict):
            flat_slot_tasks = []
            for ed_key, ed_task_list in slot_workload_ed.items():
                for t_item in ed_task_list:
                    if isinstance(t_item, dict):
                        flat_slot_tasks.append(LRMATask.from_dict(t_item))
                    else:
                        flat_slot_tasks.append(t_item)
        else:
            flat_slot_tasks = slot_workload_ed

        env.update_time_slot(t, slot_workload_ed)

        for ed_idx in range(num_ed):
            ed_raw_tasks = slot_workload_ed.get(ed_idx, slot_workload_ed.get(str(ed_idx), [])) if isinstance(slot_workload_ed, dict) else []
            ed_tasks = [LRMATask.from_dict(t_item) if isinstance(t_item, dict) else t_item for t_item in ed_raw_tasks]

            for task in ed_tasks:
                s_ed = env.get_ed_state(ed_idx, task, ed_tasks)
                
                with torch.no_grad():
                    probs_ed = actor_ed(torch.FloatTensor(s_ed).unsqueeze(0)).squeeze(0).numpy()
                    ed_action = int(np.random.choice(len(probs_ed), p=probs_ed))

                if ed_action > 0:
                    s_cloud = env.get_cloud_state(task, flat_slot_tasks)
                    with torch.no_grad():
                        probs_cloud = actor_cloud(torch.FloatTensor(s_cloud).unsqueeze(0)).squeeze(0).numpy()
                        cloud_action = int(np.random.choice(len(probs_cloud), p=probs_cloud))
                else:
                    cloud_action = 0

                res = env.step_task_offloading(ed_idx, task, ed_action, cloud_action)

                task_delays.append(res['delay'])
                task_energies.append(res['energy'])
                offload_decisions.append(1 if res['is_offloaded'] else 0)

                task_records.append({
                    'slot': t,
                    'task_id': task.task_id,
                    'ed_idx': ed_idx,
                    'task_size_bits': task.size,
                    'gpu_type': task.R,
                    'is_offloaded': res['is_offloaded'],
                    'mes_assigned': res['mes_assigned'],
                    'delay': res['delay'],
                    'energy': res['energy'],
                    'seed': seed,
                    'algorithm': algorithm,
                    'queue_type': queue_type,
                    'dataset_split': 'test'
                })

        ed_queue_history.append(float(np.mean(env.q_device)))
        mes_queue_history.append(float(np.mean(env.q_es)))

    all_task_delay = float(np.sum(task_delays))
    avg_task_delay = float(np.mean(task_delays))
    offloading_ratio = float(np.mean(offload_decisions))
    avg_energy = float(np.mean(task_energies))

    summary = {
        'algorithm': algorithm,
        'seed': seed,
        'num_ed': num_ed,
        'V': V_val,
        'queue_type': queue_type,
        'reset_enabled': reset_enabled,
        'task_arrival_rate': task_arrival_rate,
        'total_evaluated_tasks': len(task_delays),
        'all_task_completion_delay': all_task_delay,
        'avg_task_completion_delay': avg_task_delay,
        'offloading_ratio': offloading_ratio,
        'avg_energy_consumption': avg_energy,
        'ed_queue_mean_bits': float(np.mean(ed_queue_history)),
        'mes_queue_mean_bits': float(np.mean(mes_queue_history))
    }

    # Save raw CSV task records
    raw_df = pd.DataFrame(task_records)
    raw_filename = f"raw_eval_{algorithm}_N{num_ed}_V{int(V_val)}_reset{reset_enabled}_seed{seed}.csv"
    raw_path = os.path.join(EnvConfig.RAW_RESULTS_DIR, raw_filename)
    raw_df.to_csv(raw_path, index=False)

    return summary, np.array(ed_queue_history), np.array(mes_queue_history)


if __name__ == "__main__":
    summary, _, _ = evaluate_policy(algorithm='LRMA', seed=42)
    print("\n" + "=" * 60)
    print("LRMA POLICY EVALUATION SUMMARY (DYNAMIC RUN)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"{k:30s}: {v}")
    print("=" * 60)
