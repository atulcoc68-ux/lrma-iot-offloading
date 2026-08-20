import numpy as np

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig


class VirtualLevelQueue:
    """
    Virtual level queue Q_{j,r}^q in MHFQ (Paper Eq. 5).
    Tracks task entry times, arrival at top of queue, processed work, and time slice.
    """
    def __init__(self, queue_level, time_slice):
        self.queue_level = queue_level  # q \in {1, 2, 3}
        self.time_slice = time_slice    # \tau_q^{ves}
        self.queue = []

    def enqueue_task(self, task, entry_time):
        self.queue.append({
            'task': task,
            'to_time': entry_time,  # to_{j,r}^q(T_{i,k}^t)
            'top_time': 0.0,        # top_{j,r}^q(T_{i,k}^t)
            'cpu_processed': 0.0,   # \gamma_{j,c}^t(p)
            'gpu_processed': 0.0    # \gamma_{j,g}^t(p)
        })

    def total_bytes(self):
        return sum(item['task'].size for item in self.queue)

    def is_empty(self):
        return len(self.queue) == 0


class MHFQProcessor:
    """
    Multi-Level Feedback Queue for processor r at MES j (Paper Section III-A.2, Eq. 5, 14-17).
    Q_{j,r} = {Q_{j,r}^1, Q_{j,r}^2, Q_{j,r}^3} with time slices \tau_1^{ves} < \tau_2^{ves} < \tau_3^{ves}.
    """
    def __init__(self, mes_idx, gpu_type, time_slices=EnvConfig.TAU_VES):
        self.mes_idx = mes_idx
        self.gpu_type = gpu_type  # r \in {0, 1, 2, 3}
        self.time_slices = time_slices  # [0.1, 0.3, 0.6] seconds
        self.q1 = VirtualLevelQueue(1, time_slices[0])
        self.q2 = VirtualLevelQueue(2, time_slices[1])
        self.q3 = VirtualLevelQueue(3, time_slices[2])

    def get_total_backlog(self):
        return self.q1.total_bytes() + self.q2.total_bytes() + self.q3.total_bytes()

    def process_task(self, task, entry_time, f_c, f_g):
        r"""
        Simulates task execution and migration through MHFQ levels Q^1 -> Q^2 -> Q^3 (Paper Eq. 14-18).
        
        Calculates:
        - CPU processing delay D_{i,k,j}^{c,t} (Paper Eq. 14)
        - GPU processing delay D_{i,k,j}^{g,t} (Paper Eq. 15)
        - Combined processing delay D_{i,k,j}^{BS,t} = max(D^c, D^g) (Paper Eq. 16)
        - Queue waiting time W_{i,k,j}^{BS,t} (Paper Eq. 17)
        - Completion delay \aleph_{i,k}^{BS,t} (Paper Eq. 18)
        """
        # Task total computation requirements in cycles
        c_req = task.C * 1e6  # CPU cycles
        g_req = (task.G * 1e6) if task.R > 0 else 0.0  # GPU cycles

        c_time_total = c_req / max(1e6, f_c)
        g_time_total = (g_req / max(1e6, f_g)) if g_req > 0 else 0.0
        pure_compute_time = max(c_time_total, g_time_total)

        # -------------------------------------------------------------
        # Level 1 Queue Q_{j,r}^1 (Time slice \tau_1^{ves})
        # -------------------------------------------------------------
        w1 = max(0.005, self.q1.total_bytes() / (f_c / 8e6 + 1e-5))
        top1 = entry_time + w1  # top_{j,r}^1(T_{i,k}^t)
        exec1 = min(pure_compute_time, self.time_slices[0])
        rem1 = max(0.0, pure_compute_time - exec1)

        if rem1 <= 1e-6:
            # Task completed in Q^1 (SC^{t,1} = 0)
            d_bs = exec1
            w_bs = w1
            to1 = entry_time
            completion_delay = to1 + w_bs + d_bs
            return d_bs, w_bs, completion_delay

        # -------------------------------------------------------------
        # Level 2 Queue Q_{j,r}^2 (Time slice \tau_2^{ves})
        # Task is paused and transferred to Q^2 (to_{j,r}^2 = top1 + exec1)
        # -------------------------------------------------------------
        to2 = top1 + exec1  # to_{j,r}^2(T_{i,k}^t)
        w2 = max(0.005, self.q2.total_bytes() / (f_c / 8e6 + 1e-5))
        top2 = to2 + w2      # top_{j,r}^2(T_{i,k}^t)
        exec2 = min(rem1, self.time_slices[1])
        rem2 = max(0.0, rem1 - exec2)

        if rem2 <= 1e-6:
            # Task completed in Q^2 (SC^{t,2} = 0)
            d_bs = exec1 + exec2
            w_bs = (top1 - entry_time) + (top2 - to2)
            completion_delay = entry_time + w_bs + d_bs
            return d_bs, w_bs, completion_delay

        # -------------------------------------------------------------
        # Level 3 Queue Q_{j,r}^3 (Time slice \tau_3^{ves} - Complex queue, no further rotation)
        # -------------------------------------------------------------
        to3 = top2 + exec2  # to_{j,r}^3(T_{i,k}^t)
        w3 = max(0.005, self.q3.total_bytes() / (f_c / 8e6 + 1e-5))
        top3 = to3 + w3      # top_{j,r}^3(T_{i,k}^t)
        exec3 = rem2

        d_bs = exec1 + exec2 + exec3
        w_bs = (top1 - entry_time) + (top2 - to2) + (top3 - to3)
        completion_delay = entry_time + w_bs + d_bs
        return d_bs, w_bs, completion_delay


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

    def enqueue(self, mes_idx, task):
        r = min(int(task.R), self.num_gpu_types)
        proc = self.processors[mes_idx][r]
        proc.q1.enqueue_task(task, entry_time=0.0)

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
    Processes tasks sequentially without multi-level feedback or parallel queue slicing.
    """
    def __init__(self, num_mes=EnvConfig.NUM_MES):
        self.num_mes = num_mes
        self.queues = {i: [] for i in range(num_mes)}

    def enqueue(self, mes_idx, task):
        self.queues[mes_idx].append(task)

    def get_queue_length(self, mes_idx, gpu_type=None):
        return sum(t.size for t in self.queues[mes_idx])

    def process_offloaded_task(self, mes_idx, task, entry_time,
                               f_c=EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0,
                               f_g=EnvConfig.MES_GPU_CAPACITY):
        q_size = self.get_queue_length(mes_idx)
        w_bs = max(0.01, q_size / (f_c / 8e6 + 1e-5))
        c_time = (task.C * 1e6) / max(1e6, f_c)
        g_time = (task.G * 1e6) / max(1e6, f_g) if task.R > 0 else 0.0
        d_bs = max(c_time, g_time)
        completion_delay = entry_time + w_bs + d_bs
        return d_bs, w_bs, completion_delay


class MMCQueue:
    """
    M/M/C Queuing Framework dividing MES resources into C parallel sub-queues (Paper Section VI-A.2).
    """
    def __init__(self, num_mes=EnvConfig.NUM_MES, c_channels=4):
        self.num_mes = num_mes
        self.c_channels = c_channels
        self.queues = {i: [[] for _ in range(c_channels)] for i in range(num_mes)}

    def enqueue(self, mes_idx, task):
        lengths = [sum(t.size for t in q) for q in self.queues[mes_idx]]
        shortest_idx = int(np.argmin(lengths))
        self.queues[mes_idx][shortest_idx].append(task)

    def get_queue_length(self, mes_idx, gpu_type=None):
        return sum(sum(t.size for t in q) for q in self.queues[mes_idx])

    def process_offloaded_task(self, mes_idx, task, entry_time,
                               f_c=EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0,
                               f_g=EnvConfig.MES_GPU_CAPACITY):
        q_size = self.get_queue_length(mes_idx) / self.c_channels
        w_bs = max(0.008, q_size / (f_c / 8e6 + 1e-5))
        c_time = (task.C * 1e6) / max(1e6, f_c)
        g_time = (task.G * 1e6) / max(1e6, f_g) if task.R > 0 else 0.0
        d_bs = max(c_time, g_time)
        completion_delay = entry_time + w_bs + d_bs
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
        # Paper Eq. (21): \Delta L(Z_t) = 0.5 * (\sum q_{after}^2 - \sum q_{before}^2)
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
