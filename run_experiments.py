import os
import sys
import time
import json
import numpy as np
import pandas as pd

try:
    from src.config import EnvConfig
    from evaluate import evaluate_policy
    from train import train_lrma_agent
except ModuleNotFoundError:
    from config import EnvConfig
    from evaluate import evaluate_policy
    from train import train_lrma_agent


def run_fig4_fig5_experiments(seeds=[42], slots=EnvConfig.TOTAL_SLOTS):
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION SWEEP FOR FIGURE 4 & FIGURE 5 (Lyapunov V Parameter Sweep)")
    print("=" * 70)
    
    v_values = EnvConfig.V_SWEEP  # [1, 10, 20, 30, 40, 50, 100]
    records = []
    
    for v in v_values:
        for seed in seeds:
            print(f"--> Running LRMA with V = {v}, Seed = {seed}")
            summary, ed_q, mes_q = evaluate_policy(algorithm='LRMA', V_val=v, seed=seed, total_slots=slots)
            summary['v_value'] = v
            records.append(summary)
            
            # Save raw queue timeseries
            df_q = pd.DataFrame({'slot': np.arange(1, len(ed_q)+1), 'ed_queue_bits': ed_q, 'mes_queue_bits': mes_q})
            df_q.to_csv(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig5_queues_V{v}_seed{seed}.csv"), index=False)

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig4_fig5_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved Fig 4 & 5 summary to {summary_path}")
    return df_summary


def run_fig6_fig7_experiments(seeds=[42], slots=EnvConfig.TOTAL_SLOTS):
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION SWEEP FOR FIGURE 6 & FIGURE 7 (Parameter Reset Strategy Ablation)")
    print("=" * 70)
    
    reset_options = [True, False]
    records = []
    
    for reset in reset_options:
        label = "LRMA" if reset else "No-reset LRMA"
        for seed in seeds:
            print(f"--> Running {label} (Reset={reset}), Seed = {seed}")
            summary, ed_q, mes_q = evaluate_policy(algorithm='LRMA', reset_enabled=reset, seed=seed, total_slots=slots)
            summary['label'] = label
            records.append(summary)
            
            df_q = pd.DataFrame({'slot': np.arange(1, len(ed_q)+1), 'ed_queue_bits': ed_q, 'mes_queue_bits': mes_q})
            df_q.to_csv(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig7_queues_reset{reset}_seed{seed}.csv"), index=False)

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig6_fig7_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved Fig 6 & 7 summary to {summary_path}")
    return df_summary


def run_fig8_experiments(seeds=[42], slots=EnvConfig.TOTAL_SLOTS):
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION SWEEP FOR FIGURE 8 (MHFQ Framework Comparison)")
    print("=" * 70)
    
    queues = ['FCFS', 'MMC', 'MHFQ']
    user_scales = [20, 25, 30]
    records = []

    for n_ed in user_scales:
        for q_type in queues:
            for seed in seeds:
                print(f"--> Running Queue Scheme={q_type}, Users N={n_ed}, Seed={seed}")
                summary, ed_q, mes_q = evaluate_policy(algorithm='LRMA', num_ed=n_ed, queue_type=q_type, seed=seed, total_slots=slots)
                summary['queue_type'] = q_type
                summary['num_ed'] = n_ed
                records.append(summary)
                
                df_q = pd.DataFrame({'slot': np.arange(1, len(ed_q)+1), 'ed_queue_bits': ed_q, 'mes_queue_bits': mes_q})
                df_q.to_csv(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_{q_type}_N{n_ed}_seed{seed}.csv"), index=False)

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig8_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved Fig 8 summary to {summary_path}")
    return df_summary


def run_fig9_fig10_experiments(seeds=[42], slots=EnvConfig.TOTAL_SLOTS):
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION SWEEP FOR FIGURE 9 & FIGURE 10 (Algorithm Comparison & Queue Fluctuations)")
    print("=" * 70)
    
    algorithms = ['LRMA', 'MA3MCO', 'L-MADDPG', 'DVCCO']
    arrival_rates = [0.4, 0.6, 0.8]
    records = []

    for rate in arrival_rates:
        for alg in algorithms:
            for seed in seeds:
                print(f"--> Running Algorithm={alg}, Arrival Rate={int(rate*100)}%, Seed={seed}")
                summary, ed_q, mes_q = evaluate_policy(algorithm=alg, task_arrival_rate=rate, seed=seed, total_slots=slots)
                summary['algorithm'] = alg
                summary['task_arrival_rate_pct'] = f"{int(rate*100)}%"
                records.append(summary)
                
                df_q = pd.DataFrame({'slot': np.arange(1, len(ed_q)+1), 'ed_queue_bits': ed_q, 'mes_queue_bits': mes_q})
                df_q.to_csv(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig10_queues_{alg}_{int(rate*100)}pct_seed{seed}.csv"), index=False)

    df_summary = pd.DataFrame(records)
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig9_fig10_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved Fig 9 & 10 summary to {summary_path}")
    return df_summary


def run_all_experiments(seeds=[42, 43, 44, 45, 46], slots=EnvConfig.TOTAL_SLOTS):
    start_t = time.time()
    print("=" * 70)
    print(f"STARTING ALL MULTI-SEED ({len(seeds)} SEEDS) SIMULATION SWEEPS FOR IEEE TNSE REPRODUCTION")
    print("=" * 70)

    # Train main LRMA agent first
    train_lrma_agent(seed=seeds[0], total_slots=slots)

    run_fig4_fig5_experiments(seeds, slots)
    run_fig6_fig7_experiments(seeds, slots)
    run_fig8_experiments(seeds, slots)
    run_fig9_fig10_experiments(seeds, slots)

    elapsed = time.time() - start_t
    print("=" * 70)
    print(f"COMPLETED ALL MULTI-SEED EXPERIMENTS IN {elapsed:.2f} SECONDS")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments(seeds=[42, 43, 44, 45, 46], slots=100)
