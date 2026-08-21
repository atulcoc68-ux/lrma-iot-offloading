# MHFQ Scheduler Equivalence Validation Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems  
**Journal:** IEEE Transactions on Network Science and Engineering (TNSE), Vol. 12, No. 2, March/April 2025  
**Target:** Deterministic Scheduler Equivalence Validation between Production Code (`MHFQProcessor` in `src/lyapunov.py`) and Ground-Truth Event-Driven Reference Model  
**File Location:** `reports/mhfq_scheduler_equivalence.md`

---

## 1. Executive Summary

A deterministic MHFQ scheduler-equivalence validation was performed comparing the production `MHFQProcessor` implementation against an independent event-driven reference scheduler (`EventDrivenMHFQReference` in `tests/test_mhfq_scheduler_equivalence.py`).

Across 10 complex test scenarios—encompassing 10 concurrent arrivals, staggered arrivals, arrival during $Q^1$/$Q^2$ execution, interleaved short/long tasks, CPU-only tasks, GPU-only tasks, CPU+GPU co-processing, 4 GPU resource types ($R=0\dots3$), and 5 MES nodes—the production `MHFQProcessor` achieved **100% numerical equivalence** with the event-driven reference model.

No counterexamples were found. `src/lyapunov.py`, experimental results, and workload traces were preserved without modification.

---

## 2. Test Case Scenarios & Equivalence Results

| Case # | Test Scenario Description | Production Code Result | Event-Driven Reference Result | Equivalence Status |
| :---: | :--- | :---: | :---: | :---: |
| **Case 1** | 10 tasks arriving simultaneously at $t=0.0$ s on MES 0 | D: 0.50s, W: 0.00-2.70s, Comp: 0.50-3.20s | D: 0.50s, W: 0.00-2.70s, Comp: 0.50-3.20s | **100% MATCH** |
| **Case 2** | Tasks arriving at staggered simulation times ($t=0.0, 0.5, 1.0 \dots$) | Matches reference completion timestamps | Matches reference completion timestamps | **100% MATCH** |
| **Case 3** | Arrival at $t=0.05$ s while another task executes in $Q^1$ | Start $Q^1$: 0.10s, Comp: 0.25s | Start $Q^1$: 0.10s, Comp: 0.25s | **100% MATCH** |
| **Case 4** | Arrival at $t=0.25$ s while another task executes in $Q^2$ | Start $Q^1$: 0.25s, Comp: 0.40s | Start $Q^1$: 0.25s, Comp: 0.40s | **100% MATCH** |
| **Case 5** | Interleaved short ($0.03$ s) and long ($1.50$ s) tasks | Short: 0.03s, Long: 1.50s | Short: 0.03s, Long: 1.50s | **100% MATCH** |
| **Case 6** | Pure CPU-only tasks ($R=0$) | $D_{BS} = 0.30$ s | $D_{BS} = 0.30$ s | **100% MATCH** |
| **Case 7** | GPU-requiring tasks ($R=1$) | $D_{BS} = 0.25$ s | $D_{BS} = 0.25$ s | **100% MATCH** |
| **Case 8** | Mixed CPU/GPU co-processing bottleneck $\max(D_c, D_g)$ | $D_{BS} = \max(0.30, 0.50) = 0.50$ s | $D_{BS} = \max(0.30, 0.50) = 0.50$ s | **100% MATCH** |
| **Case 9** | 4 GPU resource types ($R \in \{0, 1, 2, 3\}$) on same MES | Independent execution across $R$ | Independent execution across $R$ | **100% MATCH** |
| **Case 10** | Multiple independent MES nodes ($j \in \{0 \dots 4\}$) | Independent execution across $j$ | Independent execution across $j$ | **100% MATCH** |

---

## 3. Verified Scheduling Properties

1. **FIFO Ordering**: Preserved 100% across all level queues.
2. **Resource Non-Overlapping**: Server availability timelines (`avail_q1`, `avail_q2`, `avail_q3`) strictly prevent overlapping execution on the same resource.
3. **Time Slice Limits**: $Q^1$ service never exceeds $0.1$ s, $Q^2$ service never exceeds $0.3$ s, and $Q^3$ processes remaining work.
4. **Co-Processing Bottleneck**: $D_{BS} = \max(D_{CPU}, D_{GPU})$ holds across all heterogeneous tasks.
5. **Delay Accounting Identity**: $\text{completion\_delay} = \text{completion\_time} - \text{arrival\_time} = t_{trans} + W_{BS} + D_{BS}$ holds 100%.
6. **Resource Concurrency**: CPU-only ($R=0$) and GPU ($R=1, 2, 3$) resource paths run concurrently without mutual blocking.

---

## 4. Conclusion & Preservation Decision

The mathematical timeline model implemented in `MHFQProcessor` is **fully numerically equivalent** to an event-driven discrete MHFQ scheduler. 

### Decision:
**PRESERVE `src/lyapunov.py` WITHOUT MODIFICATION.**  
The current production MHFQ implementation is correct, fast, deterministic, and 100% consistent with the paper's scheduling formulations.
