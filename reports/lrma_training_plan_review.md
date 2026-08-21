# LRMA Training Implementation Plan Review

## 1. Executive Verdict

**B. APPROVED WITH CORRECTIONS**

**Rationale**: The proposed implementation plan (`reports/lrma_training_implementation_plan.md`) correctly identifies the major algorithmic mismatch in `run_colab_experiment.py` (replacing PPO clipped surrogate loss with MADRL CTDE policy gradient, reinstating target networks, soft updates, replay buffers, and parameter resetting). However, the plan inherits three unverified assumptions from legacy code (ED action space dimension $N+1$ instead of paper binary offload intent Eq. 36, restricting parameter reset strictly to the final layer rather than full primary network, and unlabelled batch size defaults). These must be corrected prior to code execution.

---

## 2. Paper-Supported Architecture

The paper specifies a **Multi-Agent Deep Deterministic / Policy Gradient Reinforcement Learning (MADRL)** framework operating under **Centralized Training with Distributed Execution (CTDE)**:
- **$N$ Decentralized ED Actors**: One per IoT End Device $i \in \{1 \dots N\}$.
- **1 Centralized Cloud Actor**: Coordinates MES node allocation for offloaded tasks.
- **Centralized Joint Critic**: Evaluates joint states and joint actions $(\mathbf{S}, \mathbf{a})$ during training.
- **Primary & Target Networks**: Soft target updates ($\xi^{soft}=0.01$) for actors and critic.
- **Persistent Experience Replay**: Minibatch sampling from $\mathcal{D}_{ed}$ and $\mathcal{D}_{cloud}$.
- **Point-to-Uniform Candidate Generation**: Action quantization with $P=5$ candidate solutions.
- **Parameter Resetting**: Primary network parameter reset every $\delta^{reset}=50$ slots.
- **LSTM Workload Predictor**: Predicts future arrival state vector $\hat{\beta}^{t+1}$.

---

## 3. ED Actor Verification

- **Paper Specification**: $N$ independent ED Actors (Section V-B.1). Decentralized execution using local state $S_{i,k}(t)$.
- **Number of Actors**: $N$ actors (e.g., $N=25$).
- **Parameter Sharing**: Non-shared parameters across ED actors (each ED has its own policy $\pi_{\theta_i}$).
- **Input Dimension**: 9 dimensions ($T_{i,k}^t [4], \widetilde{T}_{i,k-}^t [2], Q_{device\_i} [1], \sum Q_{es\_j} [1], \beta_i^{t+1} [1]$).
- **Output Action Semantics (Paper Eq. 36)**: $a_{i,k}(t) = \{x_{i,k}^t, z_{i,k}^t\}$ where $x_{i,k}^t \in \{0, 1\}$ represents local execution ($0$) vs offloading intent ($1$).
- **Status**: **APPROVED WITH CORRECTION** (Correct action dimension from legacy $N+1$ to paper binary offload intent $x_{i,k}^t \in \{0, 1\}$).

---

## 4. Cloud Actor Verification

- **Paper Specification**: 1 dedicated Cloud Actor (Section V-B.1, Eq. 35 & 37).
- **Input Dimension**: 11 dimensions ($T_{i,k}^t [4], \varsigma_{k-}^t [1], \{Q_{es\_j}\}_{j=1}^M [5], \beta_{cloud}^{t+1} [1]$).
- **Output Action Semantics (Paper Eq. 37)**: $a_{i,k}^{cloud}(t) = \{y_{i,k,j}^t, z_{i,k}^t\}$, choosing MES node $j \in \{1 \dots M\}$ ($M=5$).
- **Output Dimension**: $M=5$.
- **Status**: **VERIFIED & APPROVED**.

---

## 5. State Dimension Verification

| State Vector | Paper Equation | Exact Components | Dimension |
| :--- | :--- | :--- | :---: |
| **ED Local State $S_{i,k}(t)$** | Eq. (34) | $[T_{size}, T_C, T_G, T_R, \text{pending\_count}, \text{pending\_size}, Q_{device\_i}, \sum Q_{es\_j}, \beta_i^{t+1}]$ | **9** |
| **Cloud State $S_{i,k}^{cloud}(t)$** | Eq. (35) | $[T_{size}, T_C, T_G, T_R, \text{offloaded\_count}, Q_{es\_1} \dots Q_{es\_5}, \beta_{cloud}^{t+1}]$ | **11** |
| **Joint State $\mathbf{S}(t)$** | Section V-B.3 | Concatenation of $N$ ED states + Cloud state ($N \times 9 + 11$) | **236** ($N=25$) |

- **Status**: **VERIFIED & APPROVED**.

---

## 6. Action Dimension Verification

| Action Vector | Paper Definition | Current Code | Proposed Plan | Corrected Code | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **ED Action $a_i(t)$** | Eq. (36): $x_{i,k}^t \in \{0, 1\}$ (Local vs Offload) | Scalar $0..5$ | `DRLActor(9, N+1)` | Discrete choice $\{0, 1\}$ (Dim 2) | **CORRECTION REQUIRED** |
| **Cloud Action $a_{cloud}(t)$** | Eq. (37): $y_{i,k,j}^t \in \{1..M\}$ (MES node) | Embedded | `DRLActor(11, M)` | Discrete choice $\{0..M-1\}$ (Dim 5) | **VERIFIED** |
| **Joint Action $\mathbf{a}(t)$** | Section V-B.3: Concatenation of $a_1 \dots a_N, a_{cloud}$ | $N/A$ | Concatenated | Concatenated vector | **VERIFIED** |

- **Status**: **APPROVED WITH CORRECTION**.

---

## 7. Centralized Critic Verification

- **Paper Formulation**: Centralized Critic $Q(\mathbf{S}, \mathbf{a}; \phi)$ under CTDE (Section V-B.3, Eq. 44).
- **Input**: Joint state $\mathbf{S}$ (dim 236) concatenated with joint action $\mathbf{a}$.
- **Output**: Single scalar $Q$-value $\mathbb{R}$.
- **Evaluates**: $Q(\mathbf{S}, \mathbf{a})$, NOT $V(\mathbf{S})$.
- **Status**: **VERIFIED & APPROVED**.

---

## 8. Actor Objective Verification

- **Paper Policy Gradient (Eq. 43)**:
  $$\nabla_{\theta_h} J(\theta_h) = \frac{1}{B} \sum_{i=1}^B \nabla_{\theta_h} \pi(a|S; \theta_h) \nabla_a Q(\mathbf{S}, \mathbf{a}; \phi) \Big|_{a=\pi(S)}$$
- **Comparison to PPO**: The paper objective is a deterministic/stochastic policy gradient evaluated directly through the Centralized Critic $\nabla_a Q(\mathbf{S}, \mathbf{a})$, NOT a PPO clipped surrogate loss $\min(r \cdot A, \text{clip}(r) \cdot A)$.
- **Status**: **VERIFIED & APPROVED** (PPO removed from LRMA path).

---

## 9. Critic Target Verification

- **Paper Critic Loss (Eq. 44)**:
  $$L(\phi) = \frac{1}{B} \sum_{i=1}^B \left( y_{target} - Q(\mathbf{S}, \mathbf{a}; \phi) \right)^2 \quad \text{where } y_{target} = r_{tot} + \gamma Q_{target}(\mathbf{S}', \mathbf{a}'; \phi_{target})$$
- **TD Target Bootstrapping**: Requires target critic network $Q_{target}$ and discount factor $\gamma = 0.99$. Current `run_colab_experiment.py` instantaneous reward target ($y = r$) is a material mismatch.
- **Status**: **VERIFIED & APPROVED**.

---

## 10. Advantage Verification

- **Paper Specification**: The paper algorithm does NOT use advantage functions ($A = r - V(s)$) or Generalized Advantage Estimation (GAE).
- **Evaluation**: Policy gradients are calculated directly against the Centralized Critic $Q(\mathbf{S}, \mathbf{a})$. Advantage normalization in `run_colab_experiment.py` is a PPO artifact and will be removed.
- **Status**: **VERIFIED & APPROVED**.

---

## 11. Replay Buffer Verification

- **Paper Semantics**: Persistent Experience Replay buffers $\mathcal{D}_i$ for ED actors and $\mathcal{D}_{cloud}$ for Cloud actor (Algorithm 1, lines 2 & 22).
- **Sampling & Persistence**: Transitions are stored continuously. Training samples random minibatches from persistent buffers. Clearing buffer after single updates is an on-policy PPO artifact to be removed.
- **Batch Size $B$**: Configured to $B=64$ (`EnvConfig.BATCH_SIZE = 64`). Marked as **PAPER UNSPECIFIED / DEFENSIBLE DEFAULT**.
- **Status**: **VERIFIED & APPROVED**.

---

## 12. Target Network Verification

- **Paper Specification (Eq. 45 & Algorithm 1 lines 1, 29)**:
  - ED Target Actors $\pi_{\theta_i}^{target}$
  - Cloud Target Actor $\pi_{\theta_{cloud}}^{target}$
  - Target Centralized Critic $Q_{\phi}^{target}$
- **Soft Update Equation**:
  $$\theta^{target} = \xi^{soft} \theta^{primary} + (1 - \xi^{soft}) \theta^{target} \quad (\xi^{soft} = 0.01)$$
- **Status**: **VERIFIED & APPROVED**.

---

## 13. Parameter Reset Verification

- **Paper Requirement (Algorithm 1 lines 26–28)**:
  - Triggered when $t \pmod{\delta^{reset}} == 0$ with $\delta^{reset} = 50$ slots.
  - Paper text: "Reset parameters of primary network $\theta_h$" to mitigate primacy bias (Nikishin et al., 2022).
- **Plan Correction**: Legacy code implemented `reset_last_layer()`. Re-initialization of primary network parameters (or final layer re-initialization as a defensible option) must be explicitly documented as an **IMPLEMENTATION ASSUMPTION**.
- **Status**: **APPROVED WITH CORRECTION**.

---

## 14. Candidate Generation Verification

- **Paper Definition (Section V-B.2)**: Point-to-uniform variation method generating $P=5$ candidate solutions $a_P^t = \{a_1(t), \dots, a_P(t)\}$.
- **Candidate Sampling**: Categorical distribution over policy output probabilities sampled $P=5$ times per task decision.
- **Status**: **VERIFIED & APPROVED**.

---

## 15. Candidate Selection Verification

- **Paper Definition**: Candidates are evaluated and selected via argmax expected reward:
  $$a^*(t) = \arg\max_{a \in a_P^t} r_{expected}(a)$$
- **Status**: **VERIFIED & APPROVED**.

---

## 16. LSTM Verification

- **Paper Specification (Eq. 41–42)**: Pre-trained `WorkloadPredictor` processing historical arrival sequence $\widetilde{T}^t$ ($l=10$) to predict future arrival vector $\hat{\beta}^{t+1}$. Loss = MSE.
- **Integration**: Predicted vector $\hat{\beta}^{t+1}$ appended as state feature to ED and Cloud states.
- **Status**: **VERIFIED & APPROVED**.

---

## 17. Reward Verification

- **Paper Equations**:
  - ED Individual Reward $r_i^t$ (Eq. 38)
  - Cloud Individual Reward $r_0^t$ (Eq. 39)
  - Comprehensive Reward $r_i^{tot}(t) = \alpha r_i^t + (1 - \alpha) r_{all}^t$ (Eq. 40) with $V=20, \alpha=0.5$.
- **Implementation**: Verified strictly implemented in `src/lyapunov.py` (`LRMARewardCalculator`).
- **Status**: **VERIFIED & APPROVED**.

---

## 18. Optimizer Verification

- **Paper Specification**: Adam optimizer for primary actor and critic networks.
- **Learning Rates**: $\eta_{actor} = 0.001$, $\eta_{critic} = 0.001$ (`EnvConfig.LR_ACTOR`, `EnvConfig.LR_CRITIC`). Marked as **PAPER UNSPECIFIED / DEFENSIBLE DEFAULT** where colab config uses $0.0003$.
- **Status**: **VERIFIED & APPROVED**.

---

## 19. Plan Assumptions That Must Be Corrected

| Plan Item | Proposed Plan Value | Paper / Corrected Value | Classification |
| :--- | :--- | :--- | :--- |
| **ED Action Space** | `DRLActor(9, N+1)` | Binary offload choice $x_{i,k}^t \in \{0, 1\}$ (Dim 2) | **INFERRED (CORRECTED)** |
| **Parameter Reset Scope** | "Final layer re-initialization" | Primary network parameter reset | **IMPLEMENTATION CHOICE** |
| **Batch Size $B=64$** | $B=64$ | Configured default in codebase | **PAPER UNSPECIFIED** |
| **Learning Rates** | $0.0003 / 0.001$ | Standard Adam learning rates | **PAPER UNSPECIFIED** |

---

## 20. Final Approved Architecture

The final approved paper-faithful architecture for implementation is:

1. **$N$ Independent ED Primary Actors**: `DRLActor(state_dim=9, action_dim=2)` (non-shared weights).
2. **$N$ Independent ED Target Actors**: Soft updated with $\xi^{soft}=0.01$.
3. **1 Cloud Primary Actor**: `DRLActor(state_dim=11, action_dim=5)`.
4. **1 Cloud Target Actor**: Soft updated with $\xi^{soft}=0.01$.
5. **1 Centralized Primary Critic**: `DRLCritic(joint_state_dim=236, joint_action_dim=55)`.
6. **1 Centralized Target Critic**: Soft updated with $\xi^{soft}=0.01$.
7. **Point-to-Uniform Candidate Generation**: $P=5$ candidates per decision, argmax expected reward selection.
8. **Persistent Experience Replay**: Minibatch sampling ($B=64$) for $\mathcal{D}_{ed}$ and $\mathcal{D}_{cloud}$.
9. **MADRL Policy Gradient & TD Loss**: Critic loss $MSE(Q, r_{tot} + \gamma Q_{target}')$, Actor loss via $\nabla_a Q$.
10. **Parameter Resetting**: Primary networks reset every $\delta^{reset}=50$ slots.
11. **PPO Isolation**: Zero PPO clipped ratio, zero GAE, zero on-policy buffer clearing in LRMA path.

---

## Final Review Verdict: **APPROVED WITH CORRECTIONS**
