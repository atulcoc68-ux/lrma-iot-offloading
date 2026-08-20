import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set publication style formatting
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig

EXPECTED_SEEDS = [42, 43, 44, 45, 46]


def validate_queue_files(file_paths, expected_seeds=EXPECTED_SEEDS):
    """
    Strictly validates a list of raw queue CSV files.
    Verifies:
    1. At least one file exists.
    2. Exactly the expected seeds are present in the list of files.
    3. No unexpected seeds are included.
    4. Every file can be loaded with pandas.
    5. Required columns exist ('slot', 'ed_queue_bits', 'mes_queue_bits').
    6. All files have the same slot/time axis order.
    7. No NaN/Inf values occur in queue columns.
    """
    if not file_paths:
        raise FileNotFoundError("Expected queue files list is empty.")

    for f in file_paths:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required raw queue file missing: {f}")

    # Validate seed matching
    extracted_seeds = []
    for f in file_paths:
        fname = os.path.basename(str(f))
        match = re.search(r'seed(\d+)', fname)
        if not match:
            raise ValueError(f"Could not parse seed from filename: {fname}")
        extracted_seeds.append(int(match.group(1)))

    if sorted(list(set(extracted_seeds))) != sorted(expected_seeds):
        raise ValueError(f"Seed mismatch! Expected unique seeds {expected_seeds}, got {set(extracted_seeds)}")

    # Load and validate content
    first_slots = None
    dfs = []
    for f in file_paths:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {f}: {e}")

        required_cols = ['slot', 'ed_queue_bits', 'mes_queue_bits']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"File {f} missing required column: '{col}'")

        if first_slots is None:
            first_slots = df['slot'].values
        else:
            if not np.array_equal(df['slot'].values, first_slots):
                raise ValueError(f"File {f} has mismatched slot axis compared to first file.")

        if df['ed_queue_bits'].isnull().any() or df['mes_queue_bits'].isnull().any():
            raise ValueError(f"File {f} contains NaN values in queue columns.")

        if np.isinf(df['ed_queue_bits']).any() or np.isinf(df['mes_queue_bits']).any():
            raise ValueError(f"File {f} contains Inf values in queue columns.")

        dfs.append(df)

    return dfs


def plot_fig4_and_fig5():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig4_fig5_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Required summary file missing: {summary_path}")

    df = pd.read_csv(summary_path)
    df_agg = df.groupby('v_value').agg({
        'all_task_completion_delay': ['mean', 'std'],
        'avg_task_completion_delay': ['mean', 'std']
    }).reset_index()

    v_vals = [1, 10, 20, 30, 40, 50, 100]
    all_mean = df_agg['all_task_completion_delay']['mean'].values
    all_std = df_agg['all_task_completion_delay']['std'].fillna(0.0).values
    avg_mean = df_agg['avg_task_completion_delay']['mean'].values
    avg_std = df_agg['avg_task_completion_delay']['std'].fillna(0.0).values

    # --- Figure 4: Bar Plot of V Impact (Mean ± Standard Deviation Error Bars) ---
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(v_vals))
    width = 0.35

    rects1 = ax1.bar(x - width/2, all_mean, width, yerr=all_std, capsize=3, label='All-task completion delay (s)', color='skyblue', edgecolor='black')
    ax1.set_xlabel('V value', fontsize=11, fontweight='bold')
    ax1.set_ylabel('All-task completion delay (s)', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in v_vals])

    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, avg_mean, width / 2.0, yerr=avg_std, capsize=3, label='Average task completion delay (s)', color='lightcoral', edgecolor='black')
    ax2.set_ylabel('Average task completion delay (s)', fontsize=10)

    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=7)

    plt.title('Fig. 4. Impact of different V values on IoT system processing capacity (Mean ± Standard Deviation)', fontsize=10)
    fig.tight_layout()
    out_fig4 = os.path.join(EnvConfig.FIGURES_DIR, 'fig4_v_impact.png')
    plt.savefig(out_fig4, dpi=300)
    plt.close()
    print(f"Saved {out_fig4} (Validated 5 Seeds)")

    # --- Figure 5: Queue Fluctuations Across V Values (Shaded Standard Deviation Variance Bands) ---
    fig, (ax_mes, ax_ed) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = ['black', 'blue', 'red', 'green', 'magenta', 'cyan', 'brown']

    total_fig5_validated = 0
    for idx, v in enumerate(v_vals):
        expected_files = [
            os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig5_queues_V{v}_seed{seed}.csv")
            for seed in EXPECTED_SEEDS
        ]
        dfs = validate_queue_files(expected_files)
        total_fig5_validated += len(dfs)

        t_steps = dfs[0]['slot'].values
        mes_matrix = np.array([df_q['mes_queue_bits'].values for df_q in dfs])
        ed_matrix = np.array([df_q['ed_queue_bits'].values for df_q in dfs])

        mes_m, mes_s = np.mean(mes_matrix, axis=0), np.std(mes_matrix, axis=0)
        ed_m, ed_s = np.mean(ed_matrix, axis=0), np.std(ed_matrix, axis=0)

        ax_mes.plot(t_steps, mes_m, label=f'V={v}', color=colors[idx % len(colors)], linewidth=1.2)
        ax_mes.fill_between(t_steps, mes_m - mes_s, mes_m + mes_s, color=colors[idx % len(colors)], alpha=0.15)

        ax_ed.plot(t_steps, ed_m, label=f'V={v}', color=colors[idx % len(colors)], linewidth=1.2)
        ax_ed.fill_between(t_steps, ed_m - ed_s, ed_m + ed_s, color=colors[idx % len(colors)], alpha=0.15)

    ax_mes.set_ylabel('Average MES awaiting task size (bit)')
    ax_mes.set_title('(a) Average MES awaiting task size (bit)')
    ax_mes.legend(loc='upper left', ncol=4, fontsize=8)
    ax_mes.grid(True, linestyle=':', alpha=0.6)

    ax_ed.set_ylabel('Average ED awaiting task size (bit)')
    ax_ed.set_xlabel('Times (s)')
    ax_ed.set_title('(b) Average ED awaiting task size (bit)')
    ax_ed.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Fig. 5. Impact of different V values on IoT system fluctuations (Mean ± Standard Deviation)', fontsize=10)
    fig.tight_layout()
    out_fig5 = os.path.join(EnvConfig.FIGURES_DIR, 'fig5_v_fluctuations.png')
    plt.savefig(out_fig5, dpi=300)
    plt.close()
    print(f"Saved {out_fig5} (Validated {total_fig5_validated} Raw Files)")


def plot_fig6_and_fig7():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig6_fig7_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Required summary file missing: {summary_path}")

    df = pd.read_csv(summary_path)
    df_agg = df.groupby('label').agg({
        'all_task_completion_delay': ['mean', 'std'],
        'avg_task_completion_delay': ['mean', 'std'],
        'offloading_ratio': ['mean', 'std']
    }).reset_index()

    categories = df_agg['label'].values
    all_m = df_agg['all_task_completion_delay']['mean'].values
    all_s = df_agg['all_task_completion_delay']['std'].fillna(0.0).values
    avg_m = df_agg['avg_task_completion_delay']['mean'].values
    avg_s = df_agg['avg_task_completion_delay']['std'].fillna(0.0).values
    off_m = df_agg['offloading_ratio']['mean'].values
    off_s = df_agg['offloading_ratio']['std'].fillna(0.0).values

    # --- Figure 6: Parameter Reset Ablation Bar Plot ---
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(categories))
    width = 0.25

    r1 = ax1.bar(x - width, all_m, width, yerr=all_s, capsize=3, label='All-task completion delay', color='sandybrown', edgecolor='black')
    r2 = ax1.bar(x, avg_m, width, yerr=avg_s, capsize=3, label='Average task completion delay', color='skyblue', edgecolor='black')
    ax1.set_ylabel('Time (s)', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontweight='bold')

    for r in r1:
        ax1.annotate(f'{r.get_height():.2f}', (r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for r in r2:
        ax1.annotate(f'{r.get_height():.2f}', (r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax2 = ax1.twinx()
    r3 = ax2.bar(x + width, off_m, width, yerr=off_s, capsize=3, label='Task offloading ratio', color='thistle', edgecolor='black')
    ax2.set_ylabel('Task offloading ratio', fontsize=10)
    ax2.set_ylim(0, 1.0)

    for r in r3:
        ax2.annotate(f'{r.get_height():.4f}', (r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    plt.title('Fig. 6. Effectiveness validation of network parameter reset strategy (Mean ± Standard Deviation)', fontsize=10)
    fig.tight_layout()
    out_fig6 = os.path.join(EnvConfig.FIGURES_DIR, 'fig6_reset_ablation.png')
    plt.savefig(out_fig6, dpi=300)
    plt.close()
    print(f"Saved {out_fig6} (Validated 5 Seeds)")

    # --- Figure 7: Queue Fluctuations for Reset vs No-Reset ---
    fig, (ax_ed, ax_mes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    files_reset = [
        os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig7_queues_resetTrue_seed{seed}.csv")
        for seed in EXPECTED_SEEDS
    ]
    files_noreset = [
        os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig7_queues_resetFalse_seed{seed}.csv")
        for seed in EXPECTED_SEEDS
    ]

    dfs_l = validate_queue_files(files_reset)
    dfs_n = validate_queue_files(files_noreset)

    t_steps = dfs_l[0]['slot'].values
    ed_l_m = np.mean([df_q['ed_queue_bits'].values for df_q in dfs_l], axis=0)
    ed_n_m = np.mean([df_q['ed_queue_bits'].values for df_q in dfs_n], axis=0)
    mes_l_m = np.mean([df_q['mes_queue_bits'].values for df_q in dfs_l], axis=0)
    mes_n_m = np.mean([df_q['mes_queue_bits'].values for df_q in dfs_n], axis=0)

    ax_ed.plot(t_steps, ed_l_m, color='red', label='LRMA algorithm', linewidth=1.2)
    ax_ed.plot(t_steps, ed_n_m, color='skyblue', label='No-reset LRMA algorithm', linewidth=1.2)
    ax_ed.set_ylabel('Average ED awaiting task size (bit)')
    ax_ed.set_title('(a) Average ED awaiting task size (bit)')
    ax_ed.legend(loc='upper right')
    ax_ed.grid(True, linestyle=':', alpha=0.6)

    ax_mes.plot(t_steps, mes_l_m, color='red', label='LRMA algorithm', linewidth=1.2)
    ax_mes.plot(t_steps, mes_n_m, color='skyblue', label='No-reset LRMA algorithm', linewidth=1.2)
    ax_mes.set_ylabel('Average MES awaiting task size (bit)')
    ax_mes.set_xlabel('Times (s)')
    ax_mes.set_title('(b) Average MES awaiting task size (bit)')
    ax_mes.legend(loc='upper left')
    ax_mes.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Fig. 7. Effectiveness validation of network parameter reset strategy.', fontsize=10)
    fig.tight_layout()
    out_fig7 = os.path.join(EnvConfig.FIGURES_DIR, 'fig7_reset_fluctuations.png')
    plt.savefig(out_fig7, dpi=300)
    plt.close()
    print(f"Saved {out_fig7} (Validated 10 Raw Files)")


def plot_fig8():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig8_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Required summary file missing: {summary_path}")

    df = pd.read_csv(summary_path)
    fig = plt.figure(figsize=(12, 8))

    ax_bar = plt.subplot(2, 2, 1)
    queues = ['FCFS', 'MMC', 'MHFQ']
    x = np.arange(len(queues))
    width = 0.25

    u20_m = [df[(df['num_ed']==20) & (df['queue_type']==q)]['avg_task_completion_delay'].mean() for q in queues]
    u20_s = [df[(df['num_ed']==20) & (df['queue_type']==q)]['avg_task_completion_delay'].std() for q in queues]
    u25_m = [df[(df['num_ed']==25) & (df['queue_type']==q)]['avg_task_completion_delay'].mean() for q in queues]
    u25_s = [df[(df['num_ed']==25) & (df['queue_type']==q)]['avg_task_completion_delay'].std() for q in queues]
    u30_m = [df[(df['num_ed']==30) & (df['queue_type']==q)]['avg_task_completion_delay'].mean() for q in queues]
    u30_s = [df[(df['num_ed']==30) & (df['queue_type']==q)]['avg_task_completion_delay'].std() for q in queues]

    ax_bar.bar(x - width, u20_m, width, yerr=np.nan_to_num(u20_s), capsize=3, label='User=20 (Avg)', color='skyblue', edgecolor='black')
    ax_bar.bar(x, u25_m, width, yerr=np.nan_to_num(u25_s), capsize=3, label='User=25 (Avg)', color='sandybrown', edgecolor='black')
    ax_bar.bar(x + width, u30_m, width, yerr=np.nan_to_num(u30_s), capsize=3, label='User=30 (Avg)', color='gray', edgecolor='black')

    ax_bar.set_ylabel('Average task completion delay (s)')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['FCFS queue', 'M/M/C queue', 'Ours (MHFQ)'], fontweight='bold')
    ax_bar.set_title('(a) Task processing capability')
    ax_bar.legend(loc='upper right', fontsize=8)

    user_configs = [(20, 2, '(b) IoT system queue fluctuations (ED=20)'),
                    (25, 3, '(c) IoT system queue fluctuations (ED=25)'),
                    (30, 4, '(d) IoT system queue fluctuations (ED=30)')]

    total_fig8_validated = 0
    for u_val, sub_idx, title in user_configs:
        ax = plt.subplot(2, 2, sub_idx)
        files_fcfs = [os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_FCFS_N{u_val}_seed{seed}.csv") for seed in EXPECTED_SEEDS]
        files_mmc = [os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_MMC_N{u_val}_seed{seed}.csv") for seed in EXPECTED_SEEDS]
        files_mhfq = [os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_MHFQ_N{u_val}_seed{seed}.csv") for seed in EXPECTED_SEEDS]

        dfs_f = validate_queue_files(files_fcfs)
        dfs_m = validate_queue_files(files_mmc)
        dfs_h = validate_queue_files(files_mhfq)

        total_fig8_validated += (len(dfs_f) + len(dfs_m) + len(dfs_h))

        df_f_m = np.mean([df_q['ed_queue_bits'].values for df_q in dfs_f], axis=0)
        df_m_m = np.mean([df_q['ed_queue_bits'].values for df_q in dfs_m], axis=0)
        df_h_m = np.mean([df_q['ed_queue_bits'].values for df_q in dfs_h], axis=0)
        t_steps = dfs_f[0]['slot'].values

        ax.plot(t_steps, df_f_m, color='steelblue', label='FCFS task tackling queue', linewidth=1.2)
        ax.plot(t_steps, df_m_m, color='lightcoral', label='M/M/C queue', linewidth=1.2)
        ax.plot(t_steps, df_h_m, color='red', label='Ours (MHFQ)', linewidth=1.2)

        ax.set_xlabel('Times (s)')
        ax.set_ylabel('Average ED awaiting task size (bit)')
        ax.set_title(title)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Fig. 8. Comparison of MHFQ framework performance.', fontsize=11, fontweight='bold')
    fig.tight_layout()
    out_fig8 = os.path.join(EnvConfig.FIGURES_DIR, 'fig8_mhfq_comparison.png')
    plt.savefig(out_fig8, dpi=300)
    plt.close()
    print(f"Saved {out_fig8} (Validated {total_fig8_validated} Raw Files)")


def plot_fig9_and_fig10():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig9_fig10_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Required summary file missing: {summary_path}")

    df = pd.read_csv(summary_path)
    algs = ['LRMA', 'MA3MCO', 'L-MADDPG', 'DVCCO']

    # --- Figure 9: All-task completion delay & Offloading ratio ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    x = np.arange(len(algs))
    width = 0.25

    del_40_m = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='40%')]['all_task_completion_delay'].mean() for a in algs]
    del_40_s = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='40%')]['all_task_completion_delay'].std() for a in algs]
    del_60_m = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='60%')]['all_task_completion_delay'].mean() for a in algs]
    del_60_s = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='60%')]['all_task_completion_delay'].std() for a in algs]
    del_80_m = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='80%')]['all_task_completion_delay'].mean() for a in algs]
    del_80_s = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='80%')]['all_task_completion_delay'].std() for a in algs]

    ax1.bar(x - width, del_40_m, width, yerr=np.nan_to_num(del_40_s), capsize=3, label='40% task generation rate', color='sandybrown', edgecolor='black')
    ax1.bar(x, del_60_m, width, yerr=np.nan_to_num(del_60_s), capsize=3, label='60% task generation rate', color='skyblue', edgecolor='black')
    ax1.bar(x + width, del_80_m, width, yerr=np.nan_to_num(del_80_s), capsize=3, label='80% task generation rate', color='gray', edgecolor='black')

    ax1.set_ylabel('All-task completion delay (s)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(algs, fontweight='bold')
    ax1.set_title('(a) All-task completion delay (s)')
    ax1.legend(loc='upper left')

    off_40_m = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='40%')]['offloading_ratio'].mean() for a in algs]
    off_40_s = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='40%')]['offloading_ratio'].std() for a in algs]
    off_60_m = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='60%')]['offloading_ratio'].mean() for a in algs]
    off_60_s = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='60%')]['offloading_ratio'].std() for a in algs]
    off_80_m = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='80%')]['offloading_ratio'].mean() for a in algs]
    off_80_s = [df[(df['algorithm']==a) & (df['task_arrival_rate_pct']=='80%')]['offloading_ratio'].std() for a in algs]

    ax2.bar(x - width, off_40_m, width, yerr=np.nan_to_num(off_40_s), capsize=3, label='40%', color='sandybrown', edgecolor='black')
    ax2.bar(x, off_60_m, width, yerr=np.nan_to_num(off_60_s), capsize=3, label='60%', color='skyblue', edgecolor='black')
    ax2.bar(x + width, off_80_m, width, yerr=np.nan_to_num(off_80_s), capsize=3, label='80%', color='gray', edgecolor='black')

    ax2.set_ylabel('Task offloading ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels(algs, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.set_title('(b) Task offloading ratio')

    fig.suptitle('Fig. 9. Impact of different algorithms on IoT processing power.', fontsize=11, fontweight='bold')
    fig.tight_layout()
    out_fig9 = os.path.join(EnvConfig.FIGURES_DIR, 'fig9_algorithm_power.png')
    plt.savefig(out_fig9, dpi=300)
    plt.close()
    print(f"Saved {out_fig9} (Validated Summary Data)")

    # --- Figure 10: Queue Fluctuations per Algorithm & Rate ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    rate_labels = ['40pct', '60pct', '80pct']
    display_rates = ['40%', '60%', '80%']
    colors = {'MA3MCO': 'orange', 'L-MADDPG': 'steelblue', 'DVCCO': 'green', 'LRMA': 'red'}

    total_fig10_validated = 0
    for col_idx, (r_lbl, d_lbl) in enumerate(zip(rate_labels, display_rates)):
        ax_ed = axes[0, col_idx]
        ax_mes = axes[1, col_idx]

        for alg in algs:
            expected_files = [
                os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig10_queues_{alg}_{r_lbl}_seed{seed}.csv")
                for seed in EXPECTED_SEEDS
            ]
            dfs = validate_queue_files(expected_files)
            total_fig10_validated += len(dfs)

            t_steps = dfs[0]['slot'].values
            ed_matrix = np.array([df_q['ed_queue_bits'].values for df_q in dfs])
            mes_matrix = np.array([df_q['mes_queue_bits'].values for df_q in dfs])

            ed_m, ed_s = np.mean(ed_matrix, axis=0), np.std(ed_matrix, axis=0)
            mes_m, mes_s = np.mean(mes_matrix, axis=0), np.std(mes_matrix, axis=0)

            ax_ed.plot(t_steps, ed_m, color=colors[alg], label=f'{alg} algorithm', linewidth=1.2)
            ax_ed.fill_between(t_steps, ed_m - ed_s, ed_m + ed_s, color=colors[alg], alpha=0.1)

            ax_mes.plot(t_steps, mes_m, color=colors[alg], label=f'{alg} algorithm', linewidth=1.2)
            ax_mes.fill_between(t_steps, mes_m - mes_s, mes_m + mes_s, color=colors[alg], alpha=0.1)

        ax_ed.set_title(f'({chr(97 + col_idx*2)}) ED queue fluctuations ({d_lbl}) (Mean ± Std)')
        ax_ed.set_ylabel('Average ED awaiting task size (bit)')
        ax_ed.legend(loc='upper left', fontsize=7)
        ax_ed.grid(True, linestyle=':', alpha=0.6)

        ax_mes.set_title(f'({chr(98 + col_idx*2)}) MES queue fluctuations ({d_lbl}) (Mean ± Std)')
        ax_mes.set_ylabel('Average MES awaiting task size (bit)')
        ax_mes.set_xlabel('Times (s)')
        ax_mes.legend(loc='upper left', fontsize=7)
        ax_mes.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Fig. 10. Comparison of IoT awaiting task curve fluctuations based on different algorithms.', fontsize=11, fontweight='bold')
    fig.tight_layout()
    out_fig10 = os.path.join(EnvConfig.FIGURES_DIR, 'fig10_algorithm_fluctuations.png')
    plt.savefig(out_fig10, dpi=300)
    plt.close()
    print(f"Saved {out_fig10} (Validated {total_fig10_validated} Raw Files)")


def main():
    os.makedirs(EnvConfig.FIGURES_DIR, exist_ok=True)
    print("=" * 70)
    print("Generating Paper Figures 4-10 (Multi-Seed Mean ± Std) Dynamically from Simulation Results")
    print("=" * 70)

    plot_fig4_and_fig5()
    plot_fig6_and_fig7()
    plot_fig8()
    plot_fig9_and_fig10()

    print("=" * 70)
    print(f"Successfully generated all multi-seed paper figures in {EnvConfig.FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
