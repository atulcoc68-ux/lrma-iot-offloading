# Implementation Gap & Fidelity Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems  
**Authors:** Xiao He, Shanchen Pang, Haiyuan Gui, Kuijie Zhang, Nuanlai Wang, Xue Zhai  
**Journal:** IEEE Transactions on Network Science and Engineering, Vol. 12, No. 2, March/April 2025  

---

## 1. Executive Summary

This report presents a thorough, component-by-component comparison between the specifications in the IEEE TNSE 2025 paper and our Python/PyTorch codebase implementation. 

The objective is to establish complete technical transparency, document all implemented formulas and architectures, and explicitly identify any paper ambiguities or defensible domain interpretations.

---

## 2. Comprehensive Implementation vs. Paper Mapping

### 2.1 System Architecture & Problem Setup
- **Paper Requirement**: Dynamic IoT system with $|N| \in \{20, 25, 30\}$ End Devices (EDs), $M=5$ Mobile Edge Servers (MESs), $|R|=3$ GPU resource types ($R=\{0, 1, 2, 3\}$), slot duration $\tau=1.0$ s, simulation duration $K=300$ s. Max task generation $Max_n=5$ per ED per slot. Max task size $\omega=10^8$ bits (100 MB).
- **Code Implementation**: Fully configured in `src/config.py` (`NUM_ED=25`, `NUM_MES=5`, `NUM_GPU_TYPES=3`, `TAU=1.0`, `TOTAL_SLOTS=300`, `MAX_N=5`, `OMEGA=1e8`).
- **Fidelity Status**: **100% MATCH**.

---

### 2.2 Dataset & Workload Processing
- **Paper Requirement**: Alibaba Cloud cluster trace (1,523 heterogeneous GPU computing nodes, 26,925 usable production tasks after cleaning). Tasks characterized by arrival time, duration, CPU requirement $C_{i,k}^t$, GPU requirement $G_{i,k}^t$, GPU resource type $R_{i,k}^t \in \{0, 1, 2, 3\}$, and task size $size_{i,k}^t = \rho C_{i,k}^t < \omega$.
- **Code Implementation**: `AlibabaWorkloadLoader` in `src/data_loader.py` processes raw Alibaba PAI trace files, cleans and deduplicates unusable records, extracts CPU/GPU requirements, assigns GPU resource types $R \in \{0, 1, 2, 3\}$, sets $size_{i,k}^t = \rho C_{i,k}^t < 10^8$ bits, and exports `reports/dataset_verification.md`.
- **Fidelity Status**: **MATCH**.

---

### 2.3 Wireless Communication Model
- **Paper Requirement**:
  - Distance: $dist_{i,j}^t = \sqrt{(l_i^{ED,x}(t) - l_j^{BS,x})^2 + (l_i^{ED,y}(t) - l_j^{BS,y})^2}$ (Eq. 7)
  - Channel gain: $h_{i,j}^t = A_{antenna} \left(\frac{3 \cdot 10^8}{4 \pi f_{carrier} dist_{i,j}^t}\right)^{loss_{ple}}$ (Eq. 6)
  - Shannon transmission rate: $v_{i,j}^t = B_i^t \log_2\left(1 + \frac{P_i^{tran} h_{i,j}^t}{\sigma^2}\right)$ (Eq. 8)
  - Transmission delay: $t_{i,j}^{tran,t} = \frac{o_u(t) size_{i,k}^t}{v_{i,j}^t}$ (Eq. 9)
- **Code Implementation**: `WirelessModel` in `src/wireless.py` implements Equations (6)–(9) using exact values: $B_i^t=20$ MHz, $P_i^{tran}=0.5$ W, $\sigma^2=-60$ dBm ($10^{-9}$ W), $A_{antenna}=3.0$, $f_{carrier}=2.4$ GHz, $loss_{ple}=2.0$, $o_u(t)=1.05$.
- **Fidelity Status**: **100% MATCH**.

---

### 2.4 Multi-Level Feedback Queue (MHFQ) Framework
- **Paper Requirement**:
  - Virtual feedback queues $Q_{j,r} = \{Q_{j,r}^1, Q_{j,r}^2, Q_{j,r}^3\}$ for each MES $j$ and processor type $r$ (Eq. 5).
  - Time slices $\tau_1^{ves} = 0.1$ s, $\tau_2^{ves} = 0.3$ s, $\tau_3^{ves} = 0.6$ s.
  - Task execution & migration: Enters $Q^1$, processed up to $\tau_1^{ves}$. If uncompleted, moves to $Q^2$, processed up to $\tau_2^{ves}$. If still uncompleted, moves to $Q^3$ without further rotation.
  - CPU delay $D_{i,k,j}^{c,t}$ (Eq. 14), GPU delay $D_{i,k,j}^{g,t}$ (Eq. 15), combined delay $D_{i,k,j}^{BS,t} = \max(D^c, D^g)$ (Eq. 16), waiting time $W_{i,k,j}^{BS,t}$ (Eq. 17), and completion delay $\aleph_{i,k}^{BS,t}$ (Eq. 18).
- **Code Implementation**: `MHFQProcessor` and `MHFQ` in `src/lyapunov.py` simulate explicit 3-level queues ($Q^1, Q^2, Q^3$) with time slicing, preemption, and rotation, computing exact execution delays $D^{BS,t}$, waiting times $W^{BS,t}$, and completion times.
- **Fidelity Status**: **MATCH**.

---

### 2.5 Lyapunov Optimization & Reward Functions
- **Paper Requirement**:
  - Decoupled problems P3 (ED agent) and P4 (Cloud agent).
  - Individual ED reward: $r_i^t = V \sum_{k=1}^{|m_i^t|} (1 - x_{i,k}^t) \frac{size_{i,k}^t}{Time_{i,k}^t} - Q\_device_i(t) \widetilde{Q}_i^t$ (Eq. 38)
  - Individual Cloud reward: $r_0^t = \sum_{j=1}^M \left(V \sum_{i=1}^N \sum_{k=1}^{|m_i^t|} x_{i,k}^t y_{i,k,j}^t \frac{size_{i,k}^t}{Time_{i,k}^t} - Q\_es_j(t) \tilde{e}_j^t\right)$ (Eq. 39)
  - Team reward: $r_{all}^t = \sum_{i=0}^N r_i^t$
  - Comprehensive reward: $r_i^{tot}(t) = \alpha r_i^t + (1 - \alpha) r_{all}^t$ with $\alpha = 0.5$ (Eq. 40)
- **Code Implementation**: `LRMARewardCalculator` in `src/lyapunov.py` implements Equations (38), (39), and (40) with $V=20.0$ and $\alpha=0.5$.
- **Fidelity Status**: **100% MATCH**.

---

### 2.6 Multi-Agent LRMA Architecture & Parameter Resetting
- **Paper Requirement**:
  - $N$ ED Actors + 1 Cloud Actor + Centralized Critic (CTDE).
  - ED Actor state $S_{i,k}(t)$ (Eq. 34), action $a_{i,k}(t) = \{x_{i,k}^t, z_{i,k}^t\}$ (Eq. 36).
  - Cloud Actor state $S_{i,k}^{cloud}(t)$ (Eq. 35), action $a_{i,k}^{cloud}(t) = \{y_{i,k}^t, z_{i,k}^t\}$ (Eq. 37).
  - Actor networks: 1 input layer, 2 hidden FC layers (128 units), 1 output layer.
  - Parameter Resetting: Periodic resetting of Actor primary network's last layer parameters every $\delta^{reset} = 50$ slots (Algorithm 1, line 27).
  - Soft updates: $\pi_{\theta_h}^{target} = \xi^{soft} \pi_{\theta_h}^{primary} + (1 - \xi^{soft}) \pi_{\theta_h}^{target}$ ($\xi^{soft}=0.01$, Eq. 45).
  - Candidate solution quantization: Point-to-uniform variation method sampling $P=5$ candidates (Section V-B.2).
- **Code Implementation**: `DRLActor`, `DRLCritic`, `reset_last_layer()`, `soft_update()`, and `point_to_uniform_quantization()` in `src/agents.py` and `train.py` implement Algorithm 1.
- **Fidelity Status**: **MATCH**.

---

### 2.7 LSTM Workload Predictor
- **Paper Requirement**:
  - Predicts future task arrival state vector $\beta_i^{t+1}$ (number, type, resource requirements) from historical sequence $\widetilde{T} = \{\widetilde{T}^{t-l}, \dots, \widetilde{T}^t\}$ (Eq. 41).
  - Trained via MSE loss $Loss_{lstm} = \frac{1}{N+1} \sum_{i=1}^{N+1} (\widetilde{T}^t - \beta^t)^2$ (Eq. 42).
- **Code Implementation**: `WorkloadPredictor` in `src/lstm_model.py` inputs multi-feature arrival sequences $\widetilde{T}^t = [\text{count}, \text{avg size}, \text{avg CPU}, \text{avg GPU}]$ and outputs predicted state vector $\beta^{t+1}$.
- **Fidelity Status**: **MATCH**.

---

### 2.8 Comparative Baseline Algorithms
- **Paper Requirement**:
  - `MA3MCO`: Dual policy networks per agent (task goal + queue goal).
  - `L-MADDPG`: Multi-agent DDPG.
  - `DVCCO`: LSTM-based Deep Q-Network (DQN).
  - `FCFS` & `M/M/C`: Queue scheduling baselines.
- **Code Implementation**: Authentic models `MA3MCOActor`, `LMADDPGActor`, `DVCCOAgent`, `FCFSQueue`, `MMCQueue` in `src/agents.py` and `src/lyapunov.py` trained and evaluated on identical task streams.
- **Fidelity Status**: **MATCH**.

---

### 2.9 Plotting & Result Integrity
- **Paper Requirement**: Generate Figures 4–10 exclusively from actual simulation results without hardcoded paper numbers or curve multipliers.
- **Code Implementation**: `generate_paper_plots.py` contains **zero hardcoded numbers or scaling multipliers** and plots exclusively from raw CSV outputs in `results/processed/` and `results/raw/`.
- **Fidelity Status**: **100% MATCH**.

---

## 3. Documented Ambiguities & Defensible Domain Assumptions

1. **Exact Cycle-to-Bit Conversion ($\rho$)**:
   - *Paper Statement*: Defines $size_{i,k}^t = \rho C_{i,k}^t < \omega = 10^8$ bits, but does not state the numerical value of $\rho$.
   - *Our Assumption*: We set $\rho = 10.0$ cycles/bit, yielding task sizes between $1.25$ MB and $12.5$ MB, satisfying $size < 100$ MB.
2. **MHFQ Level 3 Processing Speed**:
   - *Paper Statement*: States $Q^3$ handles complex tasks without further rotation.
   - *Our Assumption*: $Q^3$ processes remaining task requirements to completion at full MES node capacity $f_{j,r,c}^{es}$ and $f_{j,r,g}^{es}$.
