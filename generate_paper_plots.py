import os
import glob
import time
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


def plot_fig4_and_fig5():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig4_fig5_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found. Skipping Fig 4 & 5.")
        return

    df = pd.read_csv(summary_path)
    df_agg = df.groupby('v_value').agg({
        'all_task_completion_delay': ['mean', 'std'],
        'avg_task_completion_delay': ['mean', 'std']
    }).reset_index()

    v_vals = df_agg['v_value'].values
    all_mean = df_agg['all_task_completion_delay']['mean'].values
    all_std = df_agg['all_task_completion_delay']['std'].fillna(0.0).values
    avg_mean = df_agg['avg_task_completion_delay']['mean'].values
    avg_std = df_agg['avg_task_completion_delay']['std'].fillna(0.0).values

    # --- Figure 4: Bar Plot of V Impact (Mean ± Std Error Bars) ---
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

    plt.title('Fig. 4. Impact of different V values on IoT system processing capacity.', fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig4_v_impact.png'), dpi=300)
    plt.close()
    print("Saved fig4_v_impact.png (Multi-Seed Mean ± Std)")

    # --- Figure 5: Queue Fluctuations Across V Values (Shaded Variance Bands) ---
    fig, (ax_mes, ax_ed) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    colors = ['black', 'blue', 'red', 'green', 'magenta', 'cyan', 'brown']

    for idx, v in enumerate(v_vals):
        q_files = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig5_queues_V{v}_seed*.csv"))
        if q_files:
            dfs = [pd.read_csv(f) for f in q_files]
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

    fig.suptitle('Fig. 5. Impact of different V values on IoT system fluctuations.', fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig5_v_fluctuations.png'), dpi=300)
    plt.close()
    print("Saved fig5_v_fluctuations.png (Multi-Seed Shaded Variance)")


def plot_fig6_and_fig7():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig6_fig7_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found. Skipping Fig 6 & 7.")
        return

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

    plt.title('Fig. 6. Effectiveness validation of network parameter reset strategy.', fontsize=10)
    fig.tight_layout()
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig6_reset_ablation.png'), dpi=300)
    plt.close()
    print("Saved fig6_reset_ablation.png (Multi-Seed Mean ± Std)")

    # --- Figure 7: Queue Fluctuations for Reset vs No-Reset ---
    fig, (ax_ed, ax_mes) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    q_files_lrma = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig7_queues_resetTrue_seed*.csv"))
    q_files_noreset = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig7_queues_resetFalse_seed*.csv"))

    if q_files_lrma and q_files_noreset:
        dfs_l = [pd.read_csv(f) for f in q_files_lrma]
        dfs_n = [pd.read_csv(f) for f in q_files_noreset]
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
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig7_reset_fluctuations.png'), dpi=300)
    plt.close()
    print("Saved fig7_reset_fluctuations.png")


def plot_fig8():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig8_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found. Skipping Fig 8.")
        return

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

    for u_val, sub_idx, title in user_configs:
        ax = plt.subplot(2, 2, sub_idx)
        q_fcfs = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_FCFS_N{u_val}_seed*.csv"))
        q_mmc = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_MMC_N{u_val}_seed*.csv"))
        q_mhfq = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_MHFQ_N{u_val}_seed*.csv"))

        if q_fcfs and q_mmc and q_mhfq:
            df_f_m = np.mean([pd.read_csv(f)['ed_queue_bits'].values for f in q_fcfs], axis=0)
            df_m_m = np.mean([pd.read_csv(f)['ed_queue_bits'].values for f in q_mmc], axis=0)
            df_h_m = np.mean([pd.read_csv(f)['ed_queue_bits'].values for f in q_mhfq], axis=0)
            t_steps = pd.read_csv(q_fcfs[0])['slot'].values

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
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig8_mhfq_comparison.png'), dpi=300)
    plt.close()
    print("Saved fig8_mhfq_comparison.png")


def plot_fig9_and_fig10():
    summary_path = os.path.join(EnvConfig.PROCESSED_RESULTS_DIR, "fig9_fig10_summary.csv")
    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found. Skipping Fig 9 & 10.")
        return

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
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig9_algorithm_power.png'), dpi=300)
    plt.close()
    print("Saved fig9_algorithm_power.png")

    # --- Figure 10: Queue Fluctuations per Algorithm & Rate ---
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    rate_labels = ['40%', '60%', '80%']
    colors = {'MA3MCO': 'orange', 'L-MADDPG': 'steelblue', 'DVCCO': 'green', 'LRMA': 'red'}

    for col_idx, r_lbl in enumerate(rate_labels):
        ax_ed = axes[0, col_idx]
        ax_mes = axes[1, col_idx]

        for alg in algs:
            q_files = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig10_queues_{alg}_{r_lbl}_seed*.csv"))
            if q_files:
                dfs = [pd.read_csv(f) for f in q_files]
                t_steps = dfs[0]['slot'].values
                ed_m = np.mean([df_q['ed_queue_bits'].values for df_q in dfs], axis=0)
                mes_m = np.mean([df_q['mes_queue_bits'].values for df_q in dfs], axis=0)

                ax_ed.plot(t_steps, ed_m, color=colors[alg], label=f'{alg} algorithm', linewidth=1.2)
                ax_mes.plot(t_steps, mes_m, color=colors[alg], label=f'{alg} algorithm', linewidth=1.2)

        ax_ed.set_title(f'({chr(97 + col_idx*2)}) ED queue fluctuations ({r_lbl})')
        ax_ed.set_ylabel('Average ED awaiting task size (bit)')
        ax_ed.legend(loc='upper left', fontsize=7)
        ax_ed.grid(True, linestyle=':', alpha=0.6)

        ax_mes.set_title(f'({chr(98 + col_idx*2)}) MES queue fluctuations ({r_lbl})')
        ax_mes.set_ylabel('Average MES awaiting task size (bit)')
        ax_mes.set_xlabel('Times (s)')
        ax_mes.legend(loc='upper left', fontsize=7)
        ax_mes.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle('Fig. 10. Comparison of IoT awaiting task curve fluctuations based on different algorithms.', fontsize=11, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(EnvConfig.FIGURES_DIR, 'fig10_algorithm_fluctuations.png'), dpi=300)
    plt.close()
    print("Saved fig10_algorithm_fluctuations.png")


def main():
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
