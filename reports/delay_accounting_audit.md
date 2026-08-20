# Delay-Accounting Audit Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems (IEEE TNSE 2025)  
**Audit Target:** Task Completion Timestamp vs. Elapsed Completion Delay Accounting across `src/lyapunov.py`, `src/environment.py`, `train.py`, and `evaluate.py`.  
**File Location:** `reports/delay_accounting_audit.md`

---

## 1. Executive Summary & Root Cause Analysis

An end-to-end trace of task execution revealed a critical bug in delay accounting:
In `FCFSQueue` and `MMCQueue` inside `src/lyapunov.py`, the variable `completion_delay` was computed as:
```python
completion_delay = top + d_bs
```
where `top` is the **absolute simulation timestamp** (e.g., $t = 1200.0$ s) at which service begins, and `d_bs` is the processing duration (e.g., $0.5$ s). Consequently, `completion_delay` evaluated to **$1200.5$ s**—an absolute completion timestamp ($\aleph$) rather than the elapsed completion delay relative to task generation time ($Time_{i,k}^t = \aleph - t$).

When downstream code in `environment.py`, `train.py`, and `evaluate.py` logged `completion_delay`, absolute simulation timestamps (e.g., $1216.35$ s) were averaged instead of actual elapsed delays!

---

## 2. Answers to Audit Questions

1. **What does `top` represent?**  
   `top` (or `service_start`) represents the **absolute simulation timestamp** (in seconds from $t=0$) when server resources become free and task execution begins.

2. **What does `d_bs` represent?**  
   `d_bs` represents the **pure execution/processing duration** (in seconds) required on MES CPU/GPU resources ($\max(D_c, D_g)$).

3. **What does `w_bs` represent?**  
   `w_bs` represents the **accumulated queue waiting time** (in seconds) between queue entry $to$ and service start $top$.

4. **What does `completion_delay` currently represent in buggy code paths?**  
   In `FCFSQueue` and `MMCQueue`, `completion_delay` represented the **absolute completion timestamp** $\aleph_{BS} = top + d_bs$ (e.g., $1200.5$ s).

5. **What does the paper's $\aleph_{i,k}^{BS,t}$ represent?**  
   $\aleph_{i,k}^{BS,t}$ in Equation 18 represents the **absolute completion time** of task $k$ from ED $i$ on MES $j$.  
   The paper explicitly defines the elapsed processing/completion delay as $Time_{i,k}^t = \aleph_{i,k}^{BS,t} - t$ (Paper Section III-C.2, line following Eq. 18).

6. **Is transmission delay included in the completion delay?**  
   Yes. Offloaded task queue entry time is $to = t + t^{tran}$. Total elapsed completion delay from task generation $t$ to completion $\aleph_{BS}$ is:  
   $$Time_{i,k}^t = \aleph_{BS} - t = t^{tran} + W_{BS} + D_{BS}$$

7. **Is waiting time already included?**  
   Yes. Because $top = to + W_{BS} = t + t^{tran} + W_{BS}$, absolute completion time is $\aleph_{BS} = t + t^{tran} + W_{BS} + D_{BS}$.  
   Subtracting generation time $t$ yields $Time_{i,k}^t = t^{tran} + W_{BS} + D_{BS}$.

8. **Is `entry_time` the actual task arrival time?**  
   No. The parameter `entry_time` passed to queue processors is $to = t + t^{tran}$ (arrival at MES queue).  
   The actual task generation time at the ED is $t = \text{task.arrival\_slot}$ (or `self.current_time_slot`).

9. **Are local execution and MES execution using the same delay convention?**  
   Previously no. Local execution computed elapsed delay $d_{ed} + w_{ed}$, whereas MES execution in baseline queues returned absolute simulation timestamps $\aleph_{BS}$.

10. **Does `evaluate.py` average absolute completion timestamps or actual delays?**  
    `evaluate.py` averaged `res['delay']`. When `res['delay']` received absolute timestamps, `evaluate.py` averaged absolute timestamps instead of elapsed completion delays.

---

## 3. End-to-End Task Lifecycle Trace & Numerical Example

### Concrete Numerical Example
- **Task Generation Time ($t$)**: $20.0$ s (Slot 20)
- **Transmission Delay ($t^{tran}$)**: $0.05$ s
- **Queue Entry Time ($to$)**: $20.05$ s
- **Queue Availability / Service Start ($top$)**: $1200.00$ s (due to prior queued tasks)
- **Queue Waiting Time ($W_{BS}$)**: $top - to = 1200.00 - 20.05 = 1179.95$ s
- **Processing Duration ($D_{BS}$)**: $0.50$ s
- **Absolute Completion Time ($\aleph_{BS}$)**: $top + D_{BS} = 1200.00 + 0.50 = \mathbf{1200.50\text{ s}}$
- **Correct Elapsed Completion Delay ($Time_{i,k}^t$)**: $\aleph_{BS} - t = 1200.50 - 20.00 = \mathbf{1180.50\text{ s}}$

### Queue Server Timeline Maintenance
The queue's availability timestamp MUST continue tracking absolute completion time:
$$\text{self.avail\_time}[j] = \aleph_{BS} = 1200.50\text{ s}$$
It must **NOT** be set to completion delay ($1180.50$ s), as that would corrupt the server timeline for subsequent tasks!

---

## 4. Required Code Corrections

### Files to Modify
1. **`src/lyapunov.py`**:
   - Return both `completion_time` ($\aleph_{BS} = top + d_{bs}$) and `completion_delay` ($\aleph_{BS} - \text{task.entry\_time}$) explicitly.
   - Preserve absolute completion time $\aleph_{BS}$ for server availability timelines (`avail_q1`, `avail_q2`, `avail_q3`, `avail_time`, `channel_avail`).
2. **`src/environment.py`**:
   - Record explicit task metadata in returned result dict:
     - `arrival_time`: $t$
     - `transmission_delay`: $t^{tran}$
     - `waiting_time`: $W_{BS}$ (or $W_{ED}$)
     - `processing_time`: $D_{BS}$ (or $D_{ED}$)
     - `completion_time`: $\aleph$
     - `completion_delay`: $\aleph - t = t^{tran} + W_{BS} + D_{BS}$
3. **`tests/test_mhfq_execution.py`**:
   - Add regression tests verifying arrival $= 20.0$, completion $= 1200.5 \implies$ completion delay $= 1180.5$, and next task availability remains $1200.5$.
