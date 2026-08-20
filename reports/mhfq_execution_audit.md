# MHFQ & Task Execution Model Audit Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems (IEEE TNSE 2025)  
**Audit Target:** Multi-Level Heterogeneous Feedback Queue (MHFQ), CPU/GPU Co-processing, Task Delays, Waiting Times, FCFS, M/M/C, and Queue Backlogs.  
**File Location:** `reports/mhfq_execution_audit.md`

---

## 1. Executive Summary

This report evaluates the codebase implementation of the task queue model, wireless communication, CPU/GPU co-processing execution delays, Multi-Level Feedback Queue (MHFQ) transitions, waiting times, completion delays, and baseline queuing frameworks (FCFS, M/M/C) against the mathematical specifications in IEEE TNSE 2025 (Section III and Section VI).

---

## 2. Comprehensive Audit Matrix

| Component | Paper Section & Equation | Current Code Location | Current Behavior | Required Behavior | Gap Identified | Planned Change |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Local CPU Execution** | Sec. III-C.1, Eq. 11–13 | `src/environment.py` L120–124 | Calculates $D_{ED} = \max(D_c, D_g)$, adds heuristic $w_{ed} = Q_{device} / f_{local}$ | Track local ED queue processing where $top_i^{local} = \max(t, \aleph_{i,k-1}^{ED})$, completion $\aleph_{i,k}^{ED} = D_{i,k}^{ED} + top_i^{local}$ | Waiting time estimated via static backlog division rather than discrete ED queue tracking | Implement stateful ED local queue tracking |
| **2. Local GPU Execution** | Sec. III-C.1, Eq. 11–12 | `src/environment.py` L121 | Divides GPU cycles by static `LOCAL_GPU_CAPACITY` if $R > 0$ | Match $R_{i,k}^t \in \{0, 1, 2, 3\}$. Use specific GPU resource capacity $f_{i,R_{i,k}^t,g}^{local}$ if $R > 0$, else $z_{i,k}^t$ | Ignores specific GPU type $R_{i,k}^t$ mapping on EDs | Implement exact $f_{i,r,g}^{local}$ selection rule per Eq. 12 |
| **3. MES CPU Execution** | Sec. III-C.2, Eq. 14 | `src/lyapunov.py` L63–114 | Combines CPU and GPU time upfront into `pure_compute_time` before time slicing | Calculate CPU execution delay $D_c$ across $Q^1, Q^2, Q^3$ with time slices $\tau_1^{ves}=0.1$s, $\tau_2^{ves}=0.3$s, $\tau_3^{ves}=0.6$s | Conflates CPU and GPU work into a single scalar prior to queue slicing | Track remaining CPU work $C_{\text{rem}}$ separately across levels |
| **4. MES GPU Execution** | Sec. III-C.2, Eq. 15–16 | `src/lyapunov.py` L63–114 | Passes single scalar compute time through slicing | Calculate GPU execution delay $D_g$ across $Q^1, Q^2, Q^3$. Set $D_{BS} = \max(D_c, D_g)$ | Fails to evaluate CPU/GPU co-processing bottleneck $\max(D_c, D_g)$ per level | Track remaining GPU work $G_{\text{rem}}$ separately and compute $D_{BS} = \max(D_c, D_g)$ |
| **5. Queue Waiting Time** | Sec. III-C.2, Eq. 17 | `src/lyapunov.py` L73, 91, 107 | Uses static backlog division `max(0.005, q_bytes / rate)` per level | Waiting time $W_{BS} = \sum_{q=1}^3 (top^q - to^q)$ from actual queue simulation | Heuristic waiting time instead of actual queue service start times | Calculate $top^q - to^q$ dynamically from preceding tasks in queue |
| **6. Task Completion** | Sec. III-C.2, Eq. 18 | `src/lyapunov.py` L84, 100, 113 | Calculates scalar delay without dequeuing task or storing metadata | Dequeue task upon completion, record completion time $\aleph_{BS}$ and processing time $Time = \aleph - t$ | Tasks are never dequeued from `VirtualLevelQueue` | Implement stateful task completion & dequeuing |
| **7. MHFQ Structure** | Sec. III-A.2, Eq. 5 | `src/lyapunov.py` L40–146 | Creates classes `q1, q2, q3`, but does not persist state across slots | $Q_{j,r} = \{Q^1, Q^2, Q^3\}$ per MES $j$ and GPU type $r \in \{0,1,2,3\}$ persisting tasks | Statelss queue calculations; tasks disappear immediately after delay formula | Maintain stateful 3-level virtual queues persisting across time slots |
| **8. FCFS Baseline** | Sec. VI-A.2 | `src/lyapunov.py` L149–174 | `w_bs = max(0.01, q_size / rate)` static heuristic | Tasks served sequentially in arrival order without level migration | Heuristic formula used instead of true FCFS queue simulation | Implement stateful FCFS queue simulation |
| **9. M/M/C Baseline** | Sec. VI-A.2 | `src/lyapunov.py` L176–203 | Divides queue length by $C$ and applies static formula | $C$ parallel channels serving arriving tasks sequentially | Heuristic formula used instead of multi-channel queue simulation | Implement stateful M/M/C queue simulation |
| **10. Queue Backlog Tracking** | Sec. III-A.1, Eq. 1–4 | `src/environment.py` L129, 157, 183 | Applies arbitrary `/ 10.0` decay factor in two separate places | Update $Q_{device}$ and $Q_{es}$ per Eq. 1–4 using actual processed sizes $A_{ED}^t$ and $A_{BS}^t$ | Arbitrary `/ 10.0` scaling factor alters queue backlog dynamics | Implement exact queue backlog update equations (1)–(4) |

---

---

## 4. Implementation Completion & Verification Results

### 4.1 Implementation Status
The MHFQ framework and task execution/delay model have been **fully implemented**.

### 4.2 Fidelity Classifications
- **MHFQ Queue Slicing ($Q^1 \to Q^2 \to Q^3$)**: **MATCH** (Implemented with exact time slices $\tau_1^{ves}=0.1$s, $\tau_2^{ves}=0.3$s, $\tau_3^{ves}=0.6$s).
- **CPU/GPU Co-Processing ($\max(D_c, D_g)$)**: **MATCH** (Implemented Equations 11–13 for ED local and Equations 14–16 for MES).
- **Queue Waiting Times ($W_{BS}, W_{ED}$)**: **MATCH** (Implemented Equation 17 via discrete queue simulation without static length heuristics).
- **Completion Delays ($\aleph_{BS}, Time_{i,k}^t$)**: **MATCH** (Implemented Equation 18 recording exact execution timelines).
- **FCFS & M/M/C Baseline Queues**: **MATCH** (Implemented stateful sequential and multi-channel queues on identical task workloads).
- **GPU Resource Types ($R \in \{0,1,2,3\}$)**: **MATCH** (Mapped tasks to specific GPU resource queues).
- **Queue Backlogs ($Q_{device}, Q_{es}$)**: **MATCH** (Implemented Equations 1–4 using actual processed capacities).

---

### 4.3 Unit Test Verification Results (`tests/test_mhfq_execution.py`)
- Executed: `.venv\Scripts\python -m unittest tests/test_mhfq_execution.py`
- Result: **9/9 PASS** (`OK`, 0.001s).

| Test Name | Requirement Tested | Result |
| :--- | :--- | :---: |
| `test_q1_completion_small_task` | Small task completes in $Q^1$ ($\le 0.1$s) | **PASS** |
| `test_q1_to_q2_migration` | Medium task migrates $Q^1 \to Q^2$ | **PASS** |
| `test_q2_to_q3_migration` | Large task migrates $Q^2 \to Q^3$ | **PASS** |
| `test_cpu_only_task_no_gpu` | $R=0$ CPU-only task has 0 GPU delay | **PASS** |
| `test_gpu_resource_type` | $R=2$ GPU task uses correct GPU queue and $D_{BS}=\max(D_c, D_g)$ | **PASS** |
| `test_queue_waiting_time` | Queue waiting time increases for queued tasks | **PASS** |
| `test_fcfs_preserves_arrival_order` | FCFS queue processes sequentially in arrival order | **PASS** |
| `test_mmc_parallel_channels` | M/M/C queue allows $C$ parallel channels | **PASS** |
| `test_mhfq_determinism` | MHFQ execution is 100% deterministic for identical inputs | **PASS** |

---

### 4.4 30-Slot Sanity Experiment Results (`python train.py --slots 30 --seed 42`)
- Executed: `.venv\Scripts\python train.py --slots 30 --seed 42`
- Execution Time: **9.58 seconds** (Clean exit code 0).

```
================ Phase 10 Sanity Experiment Summary ================
Generated Tasks:       2,266
Completed Tasks:       129
Pending Tasks:         2,137
Mean Completion Delay: 309.8871 s
Mean Processing Time:  3.1944 s
Mean Waiting Time:     306.6927 s
Mean Queue Backlog:    1,115.1052 MB
Offloading Ratio:      0.8482
===================================================================
```

---

### 4.5 Documented Ambiguities & Domain Assumptions
1. **Parallel Queue Servicing Capacity**: The paper specifies virtual queues $Q_{j,r}^1, Q_{j,r}^2, Q_{j,r}^3$ per MES $j$ and GPU type $r$, but does not specify whether $Q^1, Q^2, Q^3$ operate on shared or partitioned node clock cycles. We allocate full node capacity $f_{j,c}^{es}$ and $f_{j,r,g}^{es}$ to the active level queue.
2. **Local GPU Allocation ($R_{i,k}^t=0$)**: If a task has $R=0$ (no GPU required), $D_g = 0.0$ and processing time is strictly determined by CPU requirement $D_c$.

