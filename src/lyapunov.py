from collections import deque
import numpy as np

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig


class QueuedTaskState:
    """
    Task state container retained in MHFQ level queues (Paper Section III-A.2 & III-C.2).
    Tracks entry time, level migrations, remaining work, accumulated waiting time, and completion.
    """
    def __init__(self, task, entry_time, ed_id=None):
        self.task = task
        self.task_id = str(task.task_id)
        if ed_id is not None:
            self.ed_id = int(ed_id)
        else:
            self.ed_id = int(getattr(task, 'ed_id', 0))
        self.entry_time = float(entry_time)          # Initial arrival time at MES (to_{j,r}^1)
        self.queue_entry_time = float(entry_time)    # Entry time into current level queue
        self.queue_level = 1                         # 1, 2, or 3
        
        self.remaining_cpu_cycles = float(task.C * 1e6)
        self.remaining_gpu_cycles = float(task.G * 1e6) if task.R > 0 else 0.0
        
        self.accumulated_waiting_time = 0.0
        self.service_time = 0.0
        self.completion_time = 0.0

    def __repr__(self):
        return (f"QueuedTask(ID={self.task_id}, ED={self.ed_id}, L={self.queue_level}, "
                f"RemCPU={self.remaining_cpu_cycles:.1e}, RemGPU={self.remaining_gpu_cycles:.1e}, "
                f"Wait={self.accumulated_waiting_time:.4f}s)")


class MHFQProcessor:
    """
    Multi-Level Heterogeneous Feedback Queue for processor r at MES j (Paper Section III-A.2, Eq. 5, 14-18).
    Maintains explicit task container queues Q1, Q2, Q3 with time slices \tau_1^{ves}=0.1s, \tau_2^{ves}=0.3s, \tau_3^{ves}=0.6s.
    """
    def __init__(self, mes_idx, gpu_type, time_slices=EnvConfig.TAU_VES):
        self.mes_idx = mes_idx
        self.gpu_type = gpu_type  # r \in {0, 1, 2, 3}
        self.time_slices = time_slices  # [0.1, 0.3, 0.6] seconds
        
        # Explicit task container queues for Q1, Q2, Q3 (FIFO)
        self.q1 = deque()
        self.q2 = deque()
        self.q3 = deque()
        
        # Auxiliary availability timelines for queue servers
        self.avail_q1 = 0.0
        self.avail_q2 = 0.0
        self.avail_q3 = 0.0

    def get_total_backlog(self):
        """Returns total queued workload in bits across Q1, Q2, and Q3."""
        b1 = sum(item.task.size for item in self.q1)
        b2 = sum(item.task.size for item in self.q2)
        b3 = sum(item.task.size for item in self.q3)
        return b1 + b2 + b3

    def process_task(self, task, entry_time, f_c, f_g):
        r"""
        Simulates task execution and migration through explicit MHFQ queues Q1 -> Q2 -> Q3 (Paper Eq. 14-18).
        
        Calculates:
        - CPU processing delay D_{i,k,j}^{c,t} (Paper Eq. 14)
        - GPU processing delay D_{i,k,j}^{g,t} (Paper Eq. 15)
        - Combined processing delay D_{i,k,j}^{BS,t} = max(D^c, D^g) (Paper Eq. 16)
        - Queue waiting time W_{i,k,j}^{BS,t} (Paper Eq. 17)
        - Completion delay \aleph_{i,k}^{BS,t} (Paper Eq. 18)
        """
        # 1. Instantiate Task State & Enqueue into Q1
        task_state = QueuedTaskState(task, entry_time)
        self.q1.append(task_state)

        fc_rate = max(1e6, float(f_c))
        fg_rate = max(1e6, float(f_g)) if task.R > 0 else 1.0

        # -------------------------------------------------------------
        # Level 1 Queue Q1 (Time slice \tau_1^{ves} = 0.1 s)
        # -------------------------------------------------------------
        head_q1 = self.q1.popleft()
        service_start_q1 = max(head_q1.queue_entry_time, self.avail_q1)
        w1 = service_start_q1 - head_q1.queue_entry_time
        head_q1.accumulated_waiting_time += w1

        d_c1 = head_q1.remaining_cpu_cycles / fc_rate
        d_g1 = (head_q1.remaining_gpu_cycles / fg_rate) if task.R > 0 else 0.0
        d1 = max(d_c1, d_g1)

        exec1 = min(d1, self.time_slices[0])
        head_q1.service_time += exec1

        # Work progress update during Q1 interval
        cpu_consumed1 = min(head_q1.remaining_cpu_cycles, fc_rate * exec1)
        gpu_consumed1 = min(head_q1.remaining_gpu_cycles, fg_rate * exec1) if task.R > 0 else 0.0
        head_q1.remaining_cpu_cycles = max(0.0, head_q1.remaining_cpu_cycles - cpu_consumed1)
        head_q1.remaining_gpu_cycles = max(0.0, head_q1.remaining_gpu_cycles - gpu_consumed1)

        self.avail_q1 = service_start_q1 + exec1

        if head_q1.remaining_cpu_cycles <= 1e-5 and head_q1.remaining_gpu_cycles <= 1e-5:
            # Completed in Q1
            head_q1.completion_time = service_start_q1 + exec1
            return head_q1.service_time, head_q1.accumulated_waiting_time, head_q1.completion_time

        # -------------------------------------------------------------
        # Migrate Q1 -> Q2 (Time slice \tau_2^{ves} = 0.3 s)
        # -------------------------------------------------------------
        head_q1.queue_level = 2
        head_q1.queue_entry_time = service_start_q1 + exec1
        self.q2.append(head_q1)

        head_q2 = self.q2.popleft()
        service_start_q2 = max(head_q2.queue_entry_time, self.avail_q2)
        w2 = service_start_q2 - head_q2.queue_entry_time
        head_q2.accumulated_waiting_time += w2

        d_c2 = head_q2.remaining_cpu_cycles / fc_rate
        d_g2 = (head_q2.remaining_gpu_cycles / fg_rate) if task.R > 0 else 0.0
        d2 = max(d_c2, d_g2)

        exec2 = min(d2, self.time_slices[1])
        head_q2.service_time += exec2

        # Work progress update during Q2 interval
        cpu_consumed2 = min(head_q2.remaining_cpu_cycles, fc_rate * exec2)
        gpu_consumed2 = min(head_q2.remaining_gpu_cycles, fg_rate * exec2) if task.R > 0 else 0.0
        head_q2.remaining_cpu_cycles = max(0.0, head_q2.remaining_cpu_cycles - cpu_consumed2)
        head_q2.remaining_gpu_cycles = max(0.0, head_q2.remaining_gpu_cycles - gpu_consumed2)

        self.avail_q2 = service_start_q2 + exec2

        if head_q2.remaining_cpu_cycles <= 1e-5 and head_q2.remaining_gpu_cycles <= 1e-5:
            # Completed in Q2
            head_q2.completion_time = service_start_q2 + exec2
            return head_q2.service_time, head_q2.accumulated_waiting_time, head_q2.completion_time

        # -------------------------------------------------------------
        # Migrate Q2 -> Q3 (Complex queue - process remaining work to completion)
        # -------------------------------------------------------------
        head_q2.queue_level = 3
        head_q2.queue_entry_time = service_start_q2 + exec2
        self.q3.append(head_q2)

        head_q3 = self.q3.popleft()
        service_start_q3 = max(head_q3.queue_entry_time, self.avail_q3)
        w3 = service_start_q3 - head_q3.queue_entry_time
        head_q3.accumulated_waiting_time += w3

        d_c3 = head_q3.remaining_cpu_cycles / fc_rate
        d_g3 = (head_q3.remaining_gpu_cycles / fg_rate) if task.R > 0 else 0.0
        exec3 = max(d_c3, d_g3)
        head_q3.service_time += exec3

        head_q3.remaining_cpu_cycles = 0.0
        head_q3.remaining_gpu_cycles = 0.0

        self.avail_q3 = service_start_q3 + exec3
        head_q3.completion_time = service_start_q3 + exec3

        return head_q3.service_time, head_q3.accumulated_waiting_time, head_q3.completion_time


class MHFQ:
    """
    Multi-Level Heterogeneous Feedback Queue Framework (Paper Section III-A.2, Eq 5).
    Maintains MHFQ processors across M MES nodes and |R| GPU resource types.
    """
    def __init__(self, num_mes=EnvConfig.NUM_MES, num_gpu_types=EnvConfig.NUM_GPU_TYPES):
        self.num_mes = num_mes
        self.num_gpu_types = num_gpu_types
        # Processors for each MES j and GPU type r \in {0, 1, 2, 3}
        self.processors = {
            j: {r: MHFQProcessor(j, r) for r in range(num_gpu_types + 1)}
            for j in range(num_mes)
        }

    def get_queue_length(self, mes_idx, gpu_type=0):
        r = int(gpu_type) if isinstance(gpu_type, (int, float, np.integer)) else 0
        r = min(max(0, r), self.num_gpu_types)
        return self.processors[mes_idx][r].get_total_backlog()

    def process_offloaded_task(self, mes_idx, task, entry_time,
                               f_c=EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0,
                               f_g=EnvConfig.MES_GPU_CAPACITY):
        r = min(max(0, int(task.R)), self.num_gpu_types)
        proc = self.processors[mes_idx][r]
        return proc.process_task(task, entry_time, f_c, f_g)


class FCFSQueue:
    """
    First-Come First-Served Queuing Framework for MES nodes (Paper Section VI-A.2).
    Processes tasks sequentially in arrival order using explicit FIFO queue.
    """
    def __init__(self, num_mes=EnvConfig.NUM_MES):
        self.num_mes = num_mes
        self.queues = {i: deque() for i in range(num_mes)}
        self.avail_time = {i: 0.0 for i in range(num_mes)}

    def get_queue_length(self, mes_idx, gpu_type=None):
        return sum(t.size for t in self.queues[mes_idx])

    def process_offloaded_task(self, mes_idx, task, entry_time,
                               f_c=EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0,
                               f_g=EnvConfig.MES_GPU_CAPACITY):
        self.queues[mes_idx].append(task)
        head_task = self.queues[mes_idx].popleft()

        to = float(entry_time)
        top = max(to, self.avail_time[mes_idx])
        w_bs = top - to
        
        c_time = (head_task.C * 1e6) / max(1e6, float(f_c))
        g_time = ((head_task.G * 1e6) / max(1e6, float(f_g))) if head_task.R > 0 else 0.0
        d_bs = max(c_time, g_time)
        
        completion_delay = top + d_bs
        self.avail_time[mes_idx] = completion_delay
        return d_bs, w_bs, completion_delay


class MMCQueue:
    """
    M/M/C Queuing Framework dividing MES resources into C parallel service channels (Paper Section VI-A.2).
    Maintains explicit FIFO queues per channel.
    """
    def __init__(self, num_mes=EnvConfig.NUM_MES, c_channels=4):
        self.num_mes = num_mes
        self.c_channels = c_channels
        self.channel_queues = {i: [deque() for _ in range(c_channels)] for i in range(num_mes)}
        self.channel_avail = {i: [0.0] * c_channels for i in range(num_mes)}

    def get_queue_length(self, mes_idx, gpu_type=None):
        return sum(sum(t.size for t in q) for q in self.channel_queues[mes_idx])

    def process_offloaded_task(self, mes_idx, task, entry_time,
                               f_c=EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0,
                               f_g=EnvConfig.MES_GPU_CAPACITY):
        # Assign to channel with earliest availability time / shortest queue
        chans = self.channel_avail[mes_idx]
        best_chan = int(np.argmin(chans))
        
        self.channel_queues[mes_idx][best_chan].append(task)
        head_task = self.channel_queues[mes_idx][best_chan].popleft()
        
        to = float(entry_time)
        top = max(to, self.channel_avail[mes_idx][best_chan])
        w_bs = top - to
        
        c_time = (head_task.C * 1e6) / max(1e6, float(f_c))
        g_time = ((head_task.G * 1e6) / max(1e6, float(f_g))) if head_task.R > 0 else 0.0
        d_bs = max(c_time, g_time)
        
        completion_delay = top + d_bs
        self.channel_avail[mes_idx][best_chan] = completion_delay
        return d_bs, w_bs, completion_delay


class LRMARewardCalculator:
    """
    Computes Lyapunov Queue Drift & Comprehensive Multi-Agent Rewards (Paper Section V-B.3, Eq 38-40).
    """
    def __init__(self, V_penalty=EnvConfig.V, alpha=EnvConfig.ALPHA):
        self.V = V_penalty
        self.alpha = alpha  # Paper Eq. (40)

    def calculate_drift(self, q_before, q_after):
        r"""Calculates Lyapunov drift \Delta L(Z_t) (Paper Eq 21)."""
        drift = 0.5 * (np.sum(np.square(q_after)) - np.sum(np.square(q_before)))
        return drift

    def calculate_ed_individual_reward(self, task_size, task_delay, q_device, q_tilde_i, is_offloaded):
        r"""
        Calculates individual ED agent reward r_i^t (Paper Eq 38).
        r_i^t = V * \sum_{k=1}^{|m_i^t|} (1 - x_{i,k}^t) \frac{size_{i,k}^t}{Time_{i,k}^t} - Q_device_i(t) \widetilde{Q}_i^t
        """
        if is_offloaded:
            # x_{i,k}^t = 1 -> first throughput term is 0
            return - float(q_device * q_tilde_i) / 1e12
        processing_speed = task_size / max(1e-3, task_delay)
        return self.V * (processing_speed / 1e6) - float(q_device * q_tilde_i) / 1e12

    def calculate_cloud_individual_reward(self, offloaded_tasks_info, q_es_array, e_tilde_array):
        r"""
        Calculates individual Cloud agent reward r_0^t (Paper Eq 39).
        r_0^t = \sum_{j=1}^M ( V * \sum_{i,k} x_{i,k}^t y_{i,k,j}^t \frac{size_{i,k}^t}{Time_{i,k}^t} - Q_es_j(t) \tilde{e}_j^t )
        """
        r0 = 0.0
        for mes_j, (q_es, e_tilde) in enumerate(zip(q_es_array, e_tilde_array)):
            j_speed_sum = sum(t_size / max(1e-3, t_delay) for (t_size, t_delay, assigned_j) in offloaded_tasks_info if assigned_j == mes_j)
            r0 += (self.V * (j_speed_sum / 1e6) - float(q_es * e_tilde) / 1e12)
        return r0

    def calculate_comprehensive_reward(self, r_individual, r_all):
        r"""
        Calculates comprehensive reward r_i^{tot}(t) = \alpha r_i^t + (1 - \alpha) r_{all}^t (Paper Eq 40).
        """
        return self.alpha * r_individual + (1.0 - self.alpha) * r_all
