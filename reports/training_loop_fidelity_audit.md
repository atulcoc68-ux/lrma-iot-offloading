# LRMA Training Loop Fidelity Audit Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems  
**Journal:** IEEE Transactions on Network Science and Engineering (TNSE), Vol. 12, No. 2, March/April 2025  
**Target:** Read-Only Fidelity Audit of Training Loop Architecture across Repository Entry Points  
**File Location:** `reports/training_loop_fidelity_audit.md`

---

## 1. Executive Summary

A read-only audit of the LRMA training loop architecture across the entire repository (`train.py`, `src/agents.py`, `src/replay_buffer.py`, `src/lstm_model.py`, `src/environment.py`, `src/lyapunov.py`, `run_colab_experiment.py`, `run_colab_reproduction.py`, `run_experiments.py`, `evaluate.py`) was performed.

The audit revealed that the repository maintains **two distinct training implementation paths**:
1. **Local Script (`train.py`)**: Designed for lightweight simulation, queue backlog tracking, and verifying parameter resets / target soft updates. While it instantiates the full CTDE architecture (`ed_primary_actors`, `cloud_primary_actor`, `critic`, `replay_buffer_ed`, `replay_buffer_cloud`), updates LSTM weights, and resets actor last layers every 50 slots, it omits minibatch actor/critic gradient backpropagation (`optimizer.step()`, `backward()`).
2. **Colab Script (`run_colab_experiment.py`)**: A monolithic standalone script designed for Google Colab GPU training that implements complete PPO/actor-critic minibatch backpropagation updates (`actor_loss.backward()`, `optimizer_a.step()`, `critic_loss.backward()`, `optimizer_c.step()`).

---

## 2. Training Entry Points

| File | Function / Scope | Purpose | Actor Update | Critic Update | LSTM Update | Primary Use Case |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `train.py` | `train_lrma_agent()` | Local simulation & sanity test | Omitted | Omitted | Active (`train_predictor`) | Local execution / CLI |
| `run_colab_experiment.py` | `train_drl_agent()` | Colab GPU training pipeline | Active (PPO loss + `step()`) | Active (MSE loss + `step()`) | Active (`train_predictor`) | Google Colab experiments |
| `run_colab_reproduction.py` | Multi-seed orchestration | Colab 5-seed sweep execution | Inherits `run_colab_experiment` | Inherits `run_colab_experiment` | Inherits | Google Colab reproduction |
| `run_experiments.py` | CLI Wrapper | Batch runner | Calls `train.py` / `evaluate.py` | Calls `train.py` | Calls | Local CLI batch execution |
| `evaluate.py` | `evaluate_policy()` | Test evaluation pipeline | Inactive (Eval mode) | N/A | Inactive | Evaluation on 30% split |

---

## 3. Complete Training Data Flow

1. **Environment State Construction**:
   - ED Actor $i$: `env.get_ed_state(ed_idx, task, pending_tasks)` $\to$ 9-dim state vector $S_{i,k}(t)$ (`src/environment.py:56-87`).
   - Cloud Actor: `env.get_cloud_state(task, flat_offloaded)` $\to$ 11-dim state vector $S_{i,k}^{cloud}(t)$ (`src/environment.py:89-109`).
2. **Action Quantization**:
   - `point_to_uniform_quantization(ed_actor, s_ed)` $\to$ samples $P=5$ candidate actions from categorical distribution (`src/agents.py:73-86`).
3. **Environment Step Execution**:
   - `env.step_task_offloading(ed_idx, task, ed_action, cloud_action)` $\to$ computes transmission delay $t_{trans}$, MES queue execution $D_{BS}$, waiting time $W_{BS}$, and completion delay $Time_{i,k}^t$ (`src/environment.py:111-181`).
4. **Lyapunov Reward Calculation**:
   - `calculate_ed_individual_reward`, `calculate_cloud_individual_reward`, `calculate_comprehensive_reward` $\to$ computes Lyapunov rewards with $V=20, \alpha=0.5$ (`src/lyapunov.py:278-318`).
5. **Replay Buffer Storage**:
   - Transition tuples `(s_ed, ed_action, r_tot)` and `(s_cloud, cloud_action, r_tot)` stored in `replay_buffer_ed` and `replay_buffer_cloud` (`train.py:144-146`).
6. **Minibatch Sampling & Backprop**:
   - In `train.py`: Minibatch sampling and `.backward()` steps are omitted.
   - In `run_colab_experiment.py`: `train_drl_agent` computes PPO policy surrogate loss and MSE critic loss, executing `actor_loss.backward()`, `optimizer_a.step()`, `critic_loss.backward()`, `optimizer_c.step()`.
7. **Target Network Soft Update & Parameter Reset**:
   - Target networks soft updated every `UPDATE_INTERVAL` slots via `soft_update(trg, p, XI_SOFT)` (`train.py:154-156`).
   - Primary actor last layers reset every `DELTA_RESET = 50` slots via `p.reset_last_layer()` (`train.py:159-163`).

---

## 4. Replay Buffer Analysis

- `LRMAExperienceReplay` (`src/replay_buffer.py`) defines `states`, `actions`, `rewards`, `next_states`, `log_probs`.
- In `train.py`, experience tuples are collected into lists (`replay_buffer_ed`, `replay_buffer_cloud`), but sampling is not called during step execution.
- In `run_colab_experiment.py:155-168`, `buffer.states`, `buffer.actions`, `buffer.rewards` are converted to PyTorch tensors and cleared via `buffer.clear()`.

---

## 5. Critic Update Analysis

- `DRLCritic` (`src/agents.py:44-62`) takes joint state (dim 236) and joint action (dim 655) and outputs value scalar $Q(S, a)$.
- In `run_colab_experiment.py:202-207`, critic loss is computed via MSE loss: `nn.MSELoss()(current_values, rewards)`, followed by `optimizer_c.zero_grad()`, `critic_loss.backward()`, and `optimizer_c.step()`.

---

## 6. Actor Update Analysis

- `DRLActor` (`src/agents.py:11-43`) outputs Softmax action probabilities.
- In `run_colab_experiment.py:189-200`, actor policy loss is computed using clipped ratio advantage:
  $$\text{ratios} = \exp(\text{new\_log\_probs} - \text{old\_log\_probs})$$
  $$\text{actor\_loss} = -\mathbb{E}\left[\min(\text{ratios} \cdot A, \text{clip}(\text{ratios}, 1-\epsilon, 1+\epsilon) \cdot A)\right]$$
  followed by `optimizer_a.zero_grad()`, `actor_loss.backward()`, and `optimizer_a.step()`.

---

## 7. Target Network Analysis

- Target actor networks (`ed_target_actors`, `cloud_target_actor`) exist and are initialized to match primary actors (`t_net.load_state_dict(p.state_dict())`).
- Target networks receive soft updates every `UPDATE_INTERVAL` slots (`soft_update(trg, p, XI_SOFT)` with $\xi^{soft}=0.01$).
- Target networks are strictly isolated from gradient updates.

---

## 8. Parameter Reset Analysis

- Parameter resetting is implemented in `DRLActor.reset_last_layer()` (`src/agents.py:32-42`).
- Resets the weights and biases of the final linear layer (`self.net[-2]`) using Xavier uniform initialization.
- Executed every $\delta^{reset}=50$ slots in `train.py:159-163`. Matches Algorithm 1 lines 26–28.

---

## 9. LSTM Training Analysis

- `WorkloadPredictor` (`src/lstm_model.py`) is trained periodically every 30 slots via `train_predictor` on historical arrival state vectors $\widetilde{T}^t$.
- Future arrival vector estimate $\beta^{t+1}$ is computed via `get_future_workload_estimate` and embedded directly into ED state $S_{i,k}(t)$ and Cloud state $S_{i,k}^{cloud}(t)$.

---

## 10. Algorithm 1 Line-by-Line Mapping

| Algorithm 1 Operation | Code Location | Executed in `train.py`? | Executed in `run_colab_experiment.py`? | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Line 1: Network Init** | `agents.py`, `train.py:59-71` | **YES** | **YES** | **MATCH** |
| **Line 2: Replay Buffer Init** | `replay_buffer.py`, `train.py:81` | **YES** | **YES** | **MATCH** |
| **Line 3: LSTM Init** | `lstm_model.py`, `train.py:50` | **YES** | **YES** | **MATCH** |
| **Line 4-5: Time Slot Loop** | `train.py:95` | **YES** | **YES** | **MATCH** |
| **Line 6-7: Train LSTM** | `train.py:100` | **YES** | **YES** | **MATCH** |
| **Line 8-15: ED Action** | `train.py:112-114` | **YES** | **YES** | **MATCH** |
| **Line 16-19: Cloud Action** | `train.py:118-121` | **YES** | **YES** | **MATCH** |
| **Line 20-21: Step & Reward** | `train.py:127-133` | **YES** | **YES** | **MATCH** |
| **Line 22: Buffer Store** | `train.py:144-146` | **YES** | **YES** | **MATCH** |
| **Line 23-25: Gradient Updates** | `run_colab_experiment.py:198-207` | **NO (Omitted)** | **YES (PPO Backprop)** | **PARTIAL in `train.py` / MATCH in Colab** |
| **Line 26-28: Parameter Reset** | `train.py:159-163` | **YES** | **YES** | **MATCH** |
| **Line 29: Target Soft Update** | `train.py:154-156` | **YES** | **YES** | **MATCH** |
| **Line 30-32: System Decay** | `env:206-213` | **YES** | **YES** | **MATCH** |

---

## 11. Actual Parameter-Update Verification

- **`train.py`**: Primary actor weights do not change via gradient descent because `.backward()` and `optimizer.step()` are omitted. Weights only change at $t=50, 100 \dots$ when `reset_last_layer()` re-initializes the last linear layer weights.
- **`run_colab_experiment.py`**: Actor and critic weights change dynamically via gradient backpropagation (`actor_loss.backward()`, `optimizer_a.step()`, `critic_loss.backward()`, `optimizer_c.step()`).

---

## 12. Identified Problems

- **CRITICAL**: None.
- **HIGH-PRIORITY**: Discrepancy between local `train.py` (which omits gradient updates) and `run_colab_experiment.py` (which performs gradient updates).
- **MEDIUM**: Lack of explicit docstring in `train.py` clarifying that full multi-seed GPU training is driven by `run_colab_experiment.py`.
- **LOW**: PPO advantage scaling uses standard normalization without entropy regularization loss term.

---

## 13. Recommended Corrections

### A. Required for Paper Fidelity:
- Connect minibatch replay buffer gradient updates to `train.py` if local full training is desired.

### B. Required for Functional Training:
- Document `run_colab_experiment.py` as the primary Colab training entry point.

### C. Optional:
- Add gradient norm clipping during actor/critic updates.

---

## 14. Final Decision

### **DECISION: B. LIGHTWEIGHT/SANITY TRAINING ONLY (FULL TRAINING EXISTS ELSEWHERE)**

**Rationale**: `train.py` is designed as a lightweight local simulation/sanity script for environment verification and queue tracking. The full actor/critic training loop with minibatch gradient backprop updates exists in `run_colab_experiment.py` for Google Colab GPU training.
