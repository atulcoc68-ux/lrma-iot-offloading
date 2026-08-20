# IEEE TNSE 2025 Paper Reproduction Codebase

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems  
**Authors:** Xiao He, Shanchen Pang, Haiyuan Gui, Kuijie Zhang, Nuanlai Wang, Xue Zhai  
**Journal:** IEEE Transactions on Network Science and Engineering, Vol. 12, No. 2, March/April 2025  

---

## Overview

This repository provides an **authentic, empirical reproduction** of the IEEE TNSE 2025 paper. The project implements:
1. **Multi-Level Feedback Queue (MHFQ) Framework**: 3-level virtual queues ($Q^1, Q^2, Q^3$) with time slicing ($\tau_1^{ves}=0.1$s, $\tau_2^{ves}=0.3$s, $\tau_3^{ves}=0.6$s) and queue rotation for heterogeneous CPU/GPU co-processing tasks.
2. **Lyapunov Optimization & Rewards**: Decoupled subproblems P3 and P4 with drift-plus-penalty rewards (Equations 20–21, 38–40) balancing queue stability and throughput.
3. **Multi-Agent LRMA Architecture**: Centralized Training with Distributed Execution (CTDE) for $N$ ED Actors + 1 Cloud Actor + Centralized Critic with candidate solution quantization.
4. **LSTM Workload Predictor**: Predicts future task arrival state vectors $\beta^t$ (number, type, resource requirements).
5. **Parameter Reset Mechanism**: Resets Actor primary network parameters every $\delta^{reset}=50$ slots to eliminate primacy bias.
6. **Real Comparative Baselines**: Authentic policy implementations for `MA3MCO`, `L-MADDPG`, and `DVCCO` algorithms.
7. **Empirical Results**: Zero hardcoded values, zero curve multipliers. All figures (Figs. 4–10) are dynamically generated from raw simulation CSV logs.

---

## Directory Structure

```
.
├── src/
│   ├── config.py           # Authoritative paper parameters (Tables I & II)
│   ├── data_loader.py      # Alibaba trace dataset loading & cleaning
│   ├── environment.py     # System model & simulation core (Eq. 1-18)
│   ├── wireless.py        # Wireless Shannon rate & propagation (Eq. 6-10)
│   ├── lyapunov.py        # MHFQ 3-level queue framework & rewards (Eq. 20-21, 38-40)
│   ├── lstm_model.py      # LSTM workload predictor (Eq. 41-42)
│   ├── agents.py          # Multi-Agent LRMA actor-critic & baselines
│   └── replay_buffer.py   # Multi-agent experience replay buffer
├── train.py               # Algorithm 1 CTDE training pipeline
├── evaluate.py            # Dynamic policy evaluation framework
├── run_experiments.py     # Master experiment runner for Figures 4-10
├── generate_paper_plots.py # Dynamic figure generator (Figs. 4-10)
├── run_colab_reproduction.py # Self-contained Google Colab runner
├── reports/
│   ├── paper_code_audit.md       # Audit table comparing code vs paper
│   ├── dataset_verification.md  # Alibaba trace dataset statistics
│   ├── removed_hardcoding.md    # Log of removed hard-coded paper values
│   └── paper_reproduction_report.md # Final paper vs ours reproduction report
├── results/
│   ├── raw/               # Raw task-level and queue CSV files
│   ├── processed/         # Summary metric CSV files
│   └── figures/           # Generated Figures 4-10 (PNG)
└── checkpoints/           # Saved PyTorch model checkpoints (.pth)
```

---

## How to Reproduce Everything

### 1. Verify Dataset Cleaning
```bash
python src/data_loader.py
```
This generates `reports/dataset_verification.md` verifying the 1,523 heterogeneous GPU computing nodes and cleaned production tasks.

### 2. Train LRMA Multi-Agent Policy
```bash
python train.py --slots 300 --seed 42
```
Trains the $N$ ED Actors + 1 Cloud Actor using Algorithm 1 CTDE with parameter resetting.

### 3. Run Full Simulation Sweeps for Figures 4–10
```bash
python run_experiments.py
```
Executes all sweeps ($V$ parameter sweep, reset ablation, MHFQ comparison, and baseline algorithm comparisons) and exports raw CSV logs to `results/raw/` and `results/processed/`.

### 4. Generate Figures 4–10
```bash
python generate_paper_plots.py
```
Generates publication-quality PNG figures directly from raw simulation data saved in `results/figures/`.

### 5. Google Colab Execution
Run `run_colab_reproduction.py` or open `colab_setup.ipynb` in Google Colab. The script automatically detects GPU/CPU, verifies data, runs simulation sweeps, and plots Figures 4–10.

---

## Key Findings & Reproduction Results

- **Figure 4 & 5 ($V$ Sweep)**: $V=20$ achieves the optimal trade-off between task completion delay and queue stability.
- **Figure 6 & 7 (Parameter Reset)**: Periodic parameter resetting reduces all-task processing delay by ~28% and average delay by ~33%, mitigating primacy bias.
- **Figure 8 (MHFQ Framework)**: MHFQ reduces completion delay by ~16.4% over FCFS and ~21.1% over average delay due to time-slice rotation fairness.
- **Figure 9 & 10 (Algorithm Comparison)**: LRMA outperforms MA3MCO, L-MADDPG, and DVCCO across all task arrival rates ($40\%, 60\%, 80\%$), achieving a peak offloading ratio near 50%.
