import math
import random
import numpy as np
from collections import deque

class Task:
    """Represents an online heterogeneous task."""
    def __init__(self, task_id, ed_id, size_bits, cpu_cycles, gpu_cycles, gpu_type):
        self.task_id = task_id
        self.ed_id = ed_id
        self.size_bits = size_bits
        self.cpu_cycles = cpu_cycles
        self.gpu_cycles = gpu_cycles
        self.gpu_type = gpu_type

class MHFQServerEngine:
    """
    Cloud-based Multi-Level Heterogeneous Feedback Queue (MHFQ) framework.
    Manages virtual rotation queues Q^1, Q^2, Q^3 for each processor type.
    """
    def __init__(self, num_gpu_types=3, tau_ves=[0.1, 0.3, 0.6]):
        self.num_gpu_types = num_gpu_types
        self.tau_ves = tau_ves  # Time slices tau_1, tau_2, tau_3
        self.queues = {r: [deque(), deque(), deque()] for r in range(1, num_gpu_types + 1)}

    def enqueue_task(self, task, assigned_gpu_type):
        g_type = task.gpu_type if task.gpu_type > 0 else assigned_gpu_type
        self.queues[g_type][0].append(task)

    def process_time_slot(self, cpu_freq, gpu_freq):
        completed_bits = 0
        for r in self.queues:
            for level in range(3):
                time_slice = self.tau_ves[level]
                queue = self.queues[r][level]
                queue_len = len(queue)
                
                for _ in range(queue_len):
                    if not queue:
                        break
                    t = queue.popleft()
                    
                    cpu_time = t.cpu_cycles / (cpu_freq + 1e-5)
                    gpu_time = t.gpu_cycles / (gpu_freq + 1e-5)
                    total_req_time = max(cpu_time, gpu_time)
                    
                    if total_req_time <= time_slice:
                        completed_bits += t.size_bits
                    else:
                        rem_ratio = (total_req_time - time_slice) / total_req_time
                        t.cpu_cycles *= rem_ratio
                        t.gpu_cycles *= rem_ratio
                        
                        if level < 2:
                            self.queues[r][level + 1].append(t)
                        else:
                            self.queues[r][2].append(t)
                            
        return completed_bits

class IoTEnvironment:
    """Physical Layer Wireless Channel & Dynamics Model."""
    def __init__(self, num_eds=20, num_bs=5, max_task_bits=1e8):
        self.N = num_eds
        self.M = num_bs
        self.max_task_bits = max_task_bits
        
        self.ed_pos = np.random.uniform(0, 500, size=(self.N, 2))
        self.bs_pos = np.random.uniform(100, 400, size=(self.M, 2))
        
        self.f_carrier = 2.4e9     # Carrier frequency
        self.loss_ple = 3.0        # Path loss exponent
        self.p_tran = 0.2          # Transmit power (W)
        self.noise = 1e-9          # Gaussian noise power
        self.bandwidth = 20e6      # Channel bandwidth
        self.antenna_gain = 1.0

    def compute_channel_gain(self, ed_id, bs_id):
        dis = np.linalg.norm(self.ed_pos[ed_id] - self.bs_pos[bs_id])
        dis = max(dis, 1.0)
        h = self.antenna_gain * ((3e8 / (4 * math.pi * self.f_carrier * dis)) ** self.loss_ple)
        return h

    def compute_transmission_rate(self, ed_id, bs_id):
        h = self.compute_channel_gain(ed_id, bs_id)
        snr = (self.p_tran * h) / self.noise
        v = self.bandwidth * math.log2(1 + snr)
        return v
