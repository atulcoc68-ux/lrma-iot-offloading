# IEEE TNSE 2025 LRMA Paper-Faithful Training Implementation Plan

## Executive Summary

The audit of `run_colab_experiment.py` established a **Major Training-Algorithm Mismatch**. The current file uses a single-agent PPO-style clipped surrogate objective with instantaneous reward regression and an on-policy buffer.

The objective is to replace this PPO-style trainer with a **paper-faithful multi-agent LRMA training algorithm** following the IEEE TNSE 2025 paper and Algorithm 1, strictly preserving all frozen components (`src/environment.py`, `src/lyapunov.py`, `src/config.py`, `src/data_loader.py`, `generate_paper_plots.py`, workload traces, historical results, and figures).

---

## Phase 1: Paper-to-Code Component Mapping Table

| Paper Component | Current Code (`run_colab_experiment.py`) | Required Code Architecture | Status | File Location |
| :--- | :--- | :--- | :--- | :--- |
| **ED Actor** | Single global Actor (`DRLActor(11, 6)`) shared across all tasks | $N$ independent ED Actors (`DRLActor(9, N+1)`), non-shared weights | `MISMATCH` | `src/lrma_networks.py` / `src/lrma_trainer.py` |
| **Cloud Actor** | None (collapsed into single actor choice $0..5$) | 1 dedicated Cloud Actor (`DRLActor(11, M)` where $M=5$) | `MISMATCH` | `src/lrma_networks.py` / `src/lrma_trainer.py` |
| **ED State** | $1 + 5 + 4 + 1 = 11$ dim single task state | 9 dim ($T_{i,k}^t, \widetilde{T}_{i,k-}^t, Q_{device\_i}, \sum Q_{es\_j}, \beta_i^{t+1}$) per ED | `MISMATCH` | `src/lrma_trainer.py` |
| **Cloud State** | $N/A$ | 11 dim ($T_{i,k}^t, \text{offloaded\_count}, Q_{es\_1..5}, \beta_{cloud}^{t+1}$) | `MISMATCH` | `src/lrma_trainer.py` |
| **ED Action** | Scalar $0..5$ from 6 choices | $a_{i,k}(t) \in \{0, 1 \dots N\}$ discretized offloading choice per ED | `MISMATCH` | `src/lrma_networks.py` |
| **Cloud Action** | Embedded into scalar choice | $a_{cloud}(t) \in \{0 \dots M-1\}$ MES node choice for offloaded task | `MISMATCH` | `src/lrma_networks.py` |
| **Joint Action** | $N/A$ | Concatenation of $N$ ED actions + 1 Cloud action | `MISMATCH` | `src/lrma_trainer.py` |
| **Candidate Generation** | Single action Categorical sampling | Point-to-uniform variation generating $P=5$ candidate joint/individual solutions | `MISMATCH` | `src/lrma_candidates.py` |
| **Candidate Evaluation** | Direct single action execution | Evaluate $P$ candidates against expected joint reward / Q-critic | `MISMATCH` | `src/lrma_candidates.py` |
| **Expected Reward** | PPO clipped surrogate objective $\min(r\cdot A, \text{clip}(r)\cdot A)$ | Paper MADRL expected reward / Q-value policy gradient objective | `MISMATCH` | `src/lrma_trainer.py` |
| **Replay Buffer** | On-policy 64-step buffer cleared after update | Persistent experience replay ($\mathcal{D}_{ed}$ and $\mathcal{D}_{cloud}$) sampled in minibatches $B=64$ | `MISMATCH` | `src/lrma_replay.py` |
| **Critic** | Single state critic `DRLCritic(11)` $\to 1$ scalar | Centralized Critic `DRLCritic(joint_state_dim, joint_action_dim)` evaluating $(\mathbf{S}, \mathbf{a})$ | `MISMATCH` | `src/lrma_networks.py` |
| **Primary Network** | Single actor & single critic primary | $N$ ED primary actors + 1 Cloud primary actor + Primary Centralized Critic | `MISMATCH` | `src/lrma_networks.py` |
| **Target Network** | None (no target networks in `run_colab_experiment.py`) | $N$ ED target actors + 1 Cloud target actor + Target Centralized Critic | `MISMATCH` | `src/lrma_networks.py` |
| **Soft Update** | Omitted | Target soft updates: $\theta^{target} = \xi^{soft} \theta^{primary} + (1-\xi^{soft}) \theta^{target}$ ($\xi^{soft}=0.01$) | `MISMATCH` | `src/lrma_trainer.py` |
| **Parameter Reset** | Omitted | Primary network final layer re-initialization every $\delta^{reset}=50$ slots | `MISMATCH` | `src/lrma_networks.py` |
| **LSTM Integration** | 1D scalar CPU workload predictor | Pre-trained `WorkloadPredictor` predicting future arrival vector $\hat{\beta}^{t+1}$ | `PARTIAL` | `src/lstm_model.py` / `src/lrma_trainer.py` |
| **Reward** | Instantaneous Lyapunov drift-plus-penalty | Paper Eq. (38, 39, 40) individual & comprehensive reward ($V=20, \alpha=0.5$) | `MOSTLY MATCH` | `src/lyapunov.py` |
| **Adam Optimization** | Adam ($\text{lr}_a=0.0003, \text{lr}_c=0.001$) | Separate Adam optimizers for ED primary actors, Cloud primary actor, and Centralized Critic | `PARTIAL` | `src/lrma_trainer.py` |

---

## Technical Design & Architecture Overview

### 1. Modular Component Strategy

To maintain clean modularity and prevent bloating `run_colab_experiment.py`, new paper-faithful modules will be created under `src/`:
- `src/lrma_networks.py`: Multi-agent Primary/Target ED Actors, Cloud Actor, Centralized Critic, parameter reset logic.
- `src/lrma_candidates.py`: Point-to-uniform candidate generation ($P=5$), candidate evaluation, argmax selection.
- `src/lrma_replay.py`: Persistent ED and Cloud experience replay buffers with minibatch sampling semantics.
- `src/lrma_trainer.py`: Full Algorithm 1 paper-faithful training pipeline and gradient update equations.

### 2. Integration Target
Once isolated modules pass deterministic unit tests, `run_colab_experiment.py` will be converted to use the paper-faithful `src/lrma_trainer.py` implementation instead of the PPO single-agent trainer.

---

## User Review Required

> [!IMPORTANT]
> **NO CURVE FITTING / NO NUMERICAL MANIPULATION**
> - The objective is strictly mathematical and algorithmic fidelity to the IEEE TNSE 2025 paper.
> - Curves will NOT be fitted to match paper plots.
> - Frozen components (`src/environment.py`, `src/lyapunov.py`, `src/config.py`, `src/data_loader.py`, `generate_paper_plots.py`) will NOT be modified.

---

## Implementation Phases & Deliverables

### Phase 2: Implementation of Paper-Faithful Architecture
- Create `src/lrma_networks.py` ($N$ ED Actors + 1 Cloud Actor + Target Networks + Centralized Critic).
- Create `src/lrma_candidates.py` (Point-to-uniform candidate generation $P=5$, argmax selection).
- Create `src/lrma_replay.py` (Persistent transition storage for ED and Cloud, minibatch sampling $B=64$).
- Create `src/lrma_trainer.py` (Algorithm 1 time-slot loop, TD loss, policy gradient, soft updates, parameter reset every $\delta^{reset}=50$).

### Phase 3: PPO Removal from LRMA Target Path
- Isolate PPO logic entirely from the paper-faithful LRMA pipeline.

### Phase 4: Training Loop Integration
- Integrate isolated modules into `run_colab_experiment.py` while ensuring full CPU/CUDA Colab compatibility and small sanity modes.

### Phase 5 & 6: Test-First Development & Static Equation Verification
- `tests/test_lrma_training_architecture.py`: 18 architectural assertions verifying actor counts, non-shared parameters, target networks, candidate count $P$, reset behavior, replay buffers, PPO absence, frozen file safety.
- `tests/test_lrma_training_equations.py`: Independent mathematical tests verifying candidate objective, argmax selection, critic tensor shapes, soft updates, reset behavior, and deterministic gradient updates.

### Phase 7 & 8: Regression Verification & Implementation Audit Report
- Execute existing pytest suite (`test_execution_units.py`, `test_mhfq_execution.py`, `test_mhfq_scheduler_equivalence.py`, `test_plot_data_integrity.py`).
- Execute new LRMA tests.
- Produce `reports/lrma_training_implementation_audit.md`.

---

## Verification Plan

### Automated Tests
```bash
python -m pytest tests/test_execution_units.py -v
python -m pytest tests/test_mhfq_execution.py -v
python -m pytest tests/test_mhfq_scheduler_equivalence.py -v
python -m pytest tests/test_plot_data_integrity.py -v
python -m pytest tests/test_lrma_training_architecture.py -v
python -m pytest tests/test_lrma_training_equations.py -v
```

### Manual Verification
- Verify frozen components remain bit-for-bit identical via git status/diff.
- Verify no historical CSV/JSON/PNG results were deleted or altered.
