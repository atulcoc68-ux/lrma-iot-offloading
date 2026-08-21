# LRMA Training Equation Fidelity Audit Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems  
**Journal:** IEEE Transactions on Network Science and Engineering (TNSE), Vol. 12, No. 2, March/April 2025  
**Target:** Read-Only Mathematical & Equation Fidelity Audit of DRL Training Implementation in `run_colab_experiment.py`  
**File Location:** `reports/lrma_training_equation_fidelity.md`

---

## 1. Executive Summary

A read-only mathematical fidelity audit of the full training script (`run_colab_experiment.py`) was performed against the IEEE TNSE 2025 paper and Algorithm 1.

While `run_colab_experiment.py` executes successfully, converges, and produces functional plots, its underlying training equations represent a **PPO-based single-agent policy gradient implementation** rather than the paper's **MADDPG-based Multi-Agent CTDE framework**.

Key mathematical discrepancies include:
1. **PPO Clipped Surrogate Loss**: Implements `torch.clamp(ratios, 1-eps, 1+eps)` loss instead of MADDPG policy gradient.
2. **Missing TD Target Bootstrapping**: Critic target is set directly to instantaneous reward `rewards` rather than $r_{tot} + \gamma Q_{target}(S', a')$.
3. **Single-Agent Simplification**: Uses a single Actor (dim 11 $\to$ 6) and single Critic (dim 11 $\to$ 1) instead of $N$ ED Actors + 1 Cloud Actor + Centralized Joint Critic (dim 891 $\to$ 1).
4. **Buffer Clearing**: Replay buffer is cleared after every update (`on-policy` mode) rather than sampling from persistent experience replay.

---

## 2. Paper Training Algorithm

The paper specifies a Multi-Agent Deep Deterministic Policy Gradient (MADDPG) algorithm operating under Centralized Training with Distributed Execution (CTDE):
- **Critic Loss (Paper Eq. 44)**:
  $$L(\phi) = \frac{1}{B} \sum_{i=1}^B \left( y_{target} - Q(\mathbf{S}, \mathbf{a}; \phi) \right)^2 \quad \text{where } y_{target} = r_{tot} + \gamma Q_{target}(\mathbf{S}', \mathbf{a}'; \phi_{target})$$
- **Actor Policy Gradient (Paper Eq. 43)**:
  $$\nabla_{\theta_h} J(\theta_h) = \frac{1}{B} \sum_{i=1}^B \nabla_{\theta_h} \pi(a|S; \theta_h) \nabla_a Q(\mathbf{S}, \mathbf{a}; \phi) \Big|_{a=\pi(S)}$$
- **Soft Target Update (Paper Eq. 45)**:
  $$\theta_{target} = \xi^{soft} \theta_{primary} + (1 - \xi^{soft}) \theta_{target} \quad (\xi^{soft}=0.01)$$
- **Parameter Reset**: Primary network final layer weights reset every $\delta^{reset}=50$ slots.

---

## 3. Implemented Training Algorithm (`run_colab_experiment.py`)

- **Actor Loss**:
  $$\text{ratios} = \exp(\text{log\_prob}_{new} - \text{log\_prob}_{old})$$
  $$\text{surr1} = \text{ratios} \cdot A, \quad \text{surr2} = \text{clip}(\text{ratios}, 1-\epsilon, 1+\epsilon) \cdot A$$
  $$\text{actor\_loss} = -\text{mean}(\min(\text{surr1}, \text{surr2}))$$
- **Critic Loss**:
  $$\text{critic\_loss} = \text{MSELoss}(\text{critic}(\text{old\_states}), \text{rewards})$$
- **Advantage Calculation**:
  $$A_{raw} = \text{rewards} - \text{critic}(\text{old\_states}), \quad A = \frac{A_{raw} - \mu(A_{raw})}{\sigma(A_{raw}) + 1e-8}$$

---

## 4. Actor Loss Comparison

- **Paper Specification**: Policy gradient evaluated against Centralized Critic $Q(\mathbf{S}, \mathbf{a})$.
- **Code Implementation**: PPO clipped ratio surrogate loss operating on Categorical distribution log-probabilities (`run_colab_experiment.py:189-196`).
- **Match Status**: **MATERIAL MISMATCH**.

---

## 5. Critic Loss Comparison

- **Paper Specification**: Temporal Difference (TD) target bootstrapping $y_{target} = r + \gamma Q_{target}(\mathbf{S}', \mathbf{a}')$.
- **Code Implementation**: $\text{critic\_loss} = \text{MSELoss}(\text{critic}(s), \text{rewards})$. Instantaneous reward regression without target network $Q_{target}$ or discount factor $\gamma$.
- **Match Status**: **MATERIAL MISMATCH**.

---

## 6. Advantage Comparison

- **Paper Specification**: Direct Q-value evaluation $Q(\mathbf{S}, \mathbf{a})$.
- **Code Implementation**: Normalized advantage $A = (r - V(s))_{normalized}$.
- **Match Status**: **PARTIAL / MISMATCH**.

---

## 7. PPO vs LRMA Analysis

- **Finding**: The implementation uses PPO-style clipped surrogate optimization (`eps_clip = 0.2`), but the paper specifies a MADDPG-style CTDE framework.
- **Impact**: While PPO converges effectively in practice, it is algorithmically distinct from the paper's policy gradient derivation.

---

## 8. Replay/On-Policy Buffer Analysis

- **Paper Specification**: Persistent Experience Replay buffer $\mathcal{D}$ sampled randomly in minibatches of size $B=64$.
- **Code Implementation**: Buffer collects transitions over 64 steps, performs 1 gradient update, and clears the buffer via `buffer.clear()`.
- **Match Status**: **ON-POLICY ROLLOUT (MISMATCH)**.

---

## 9. CTDE Architecture Analysis

- **Paper Specification**: $N=25$ ED Actors + 1 Cloud Actor + Centralized Joint Critic (joint state dim 236, joint action dim 655).
- **Code Implementation**: Single Actor (state dim 11 $\to$ 6) + Single Critic (state dim 11 $\to$ 1).
- **Match Status**: **SINGLE-AGENT SIMPLIFICATION**.

---

## 10. Multi-Agent Update Analysis

- In `run_colab_experiment.py`, a single global actor and critic are updated. Decentralized per-ED actors from `src/agents.py` and `train.py` are not called.

---

## 11. Target Network Analysis

- Target actor/critic networks and soft updates ($\xi^{soft}=0.01$) are defined in `src/agents.py` and `train.py`, but omitted in `run_colab_experiment.py`.

---

## 12. Parameter Reset Analysis

- Parameter resetting (`reset_last_layer()`) every $\delta^{reset}=50$ slots is implemented in `src/agents.py` and `train.py`, but omitted in `run_colab_experiment.py`.

---

## 13. LSTM Integration Analysis

- `WorkloadPredictor` is pre-trained on CPU workload data and predicts $\hat{\beta}$, which is appended as the 11th feature in `state`. **FULL MATCH**.

---

## 14. Tensor Dimension Audit Table

| Component | Implemented (`run_colab_experiment.py`) | Expected (Paper CTDE) | Status |
| :--- | :--- | :--- | :---: |
| **ED Actor State** | $N/A$ (Single Actor used) | 9 dimensions ($S_{i,k}$) | **MISMATCH** |
| **Cloud Actor State** | $N/A$ (Single Actor used) | 11 dimensions ($S_{i,k}^{cloud}$) | **MISMATCH** |
| **Single Actor State** | 11 dimensions | $N/A$ | **SIMPLIFICATION** |
| **Actor Action Output** | 6 choices ($0 \dots 5$) | $N+1 = 26$ choices per ED, $M=5$ for Cloud | **SIMPLIFICATION** |
| **Critic Input** | 11 dimensions (State only) | 891 dimensions (Joint State + Joint Action) | **MISMATCH** |
| **Critic Output** | 1 scalar value | 1 scalar value | **MATCH** |

---

## 15. Exact Fidelity Gaps

### CRITICAL:
1. **Critic Target Bootstrapping**: Critic target uses $r$ directly instead of $r + \gamma Q_{target}(S', a')$.
2. **Centralized Critic Input**: Critic evaluates single state $s$ instead of joint state-action $(\mathbf{S}, \mathbf{a})$.

### HIGH-PRIORITY:
3. **PPO vs MADDPG**: PPO clipped ratio loss used instead of MADDPG policy gradient.
4. **CTDE Topology**: Multi-agent CTDE simplified into a single global actor network in `run_colab_experiment.py`.
5. **Target Networks & Parameter Reset**: Target networks and parameter reset omitted in `run_colab_experiment.py`.

### MEDIUM:
6. **On-Policy Buffer**: Buffer cleared after single update (`on-policy` rollout mode).

---

## 16. Recommended Changes

### A. Required for Paper Fidelity:
- Update `run_colab_experiment.py` (or create a Colab CTDE script) to use the full multi-agent CTDE framework (`ed_primary_actors`, `cloud_primary_actor`, `critic`) already present in `train.py` and `src/agents.py`.
- Add TD target bootstrapping: $y = r_{tot} + \gamma Q_{target}(S', a')$.
- Reinstate target soft updates ($\xi^{soft}=0.01$) and parameter resets ($\delta^{reset}=50$).

### B. Required for Functional Training:
- Maintain stable learning rates and environment interface.

---

## 17. Final Decision

### **DECISION: D. MAJOR TRAINING-ALGORITHM MISMATCH**

**Rationale**: While `run_colab_experiment.py` executes cleanly and converges functionally, its loss functions (PPO clipped surrogate loss instead of MADDPG policy gradient), critic targets (instantaneous $r$ instead of $r + \gamma Q_{target}$), and agent topology (single actor/critic instead of $N+1$ CTDE actors + joint critic) represent a major algorithmic departure from the paper's training specification.
