import unittest
import numpy as np
from collections import deque
from src.config import EnvConfig
from src.data_loader import LRMATask
from src.lyapunov import MHFQProcessor, MHFQ


class EventDrivenMHFQReference:
    """
    Independent Event-Driven Discrete-Time Reference Scheduler for MHFQ.
    Used purely as a ground-truth reference model inside unit tests to verify 
    the numerical equivalence of MHFQProcessor availability timeline math.
    """
    def __init__(self, num_mes=EnvConfig.NUM_MES, num_gpu_types=EnvConfig.NUM_GPU_TYPES):
        self.num_mes = num_mes
        self.num_gpu_types = num_gpu_types
        # Level queues per (mes_idx, gpu_type): q1, q2, q3
        self.processors = {
            (j, r): {
                'q1': deque(),
                'q2': deque(),
                'q3': deque(),
                'avail_q1': 0.0,
                'avail_q2': 0.0,
                'avail_q3': 0.0
            }
            for j in range(num_mes) for r in range(num_gpu_types + 1)
        }

    def process_task(self, mes_idx, task, entry_time,
                     f_c=EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0,
                     f_g=EnvConfig.MES_GPU_CAPACITY):
        r = min(max(0, int(task.R)), self.num_gpu_types)
        proc = self.processors[(mes_idx, r)]
        
        fc_rate = max(1e6, float(f_c))
        fg_rate = max(1e6, float(f_g)) if task.R > 0 else 1.0
        
        c_req = float(task.C * 1e6)
        g_req = float(task.G * 1e6) if task.R > 0 else 0.0
        
        to = float(entry_time)
        t_arrival = float(getattr(task, 'arrival_slot', to))
        
        # --- Level 1 Queue (tau1 = 0.1s) ---
        q1_start = max(to, proc['avail_q1'])
        w1 = q1_start - to
        
        d_c1 = c_req / fc_rate
        d_g1 = g_req / fg_rate if task.R > 0 else 0.0
        d1 = max(d_c1, d_g1)
        exec1 = min(d1, 0.1)
        
        rem_c1 = max(0.0, c_req - fc_rate * exec1)
        rem_g1 = max(0.0, g_req - fg_rate * exec1) if task.R > 0 else 0.0
        proc['avail_q1'] = q1_start + exec1
        
        if rem_c1 <= 1e-5 and rem_g1 <= 1e-5:
            comp_time = q1_start + exec1
            comp_delay = comp_time - t_arrival
            return exec1, w1, comp_time, comp_delay
        
        # --- Level 2 Queue (tau2 = 0.3s) ---
        q2_entry = q1_start + exec1
        q2_start = max(q2_entry, proc['avail_q2'])
        w2 = q2_start - q2_entry
        
        d_c2 = rem_c1 / fc_rate
        d_g2 = rem_g1 / fg_rate if task.R > 0 else 0.0
        d2 = max(d_c2, d_g2)
        exec2 = min(d2, 0.3)
        
        rem_c2 = max(0.0, rem_c1 - fc_rate * exec2)
        rem_g2 = max(0.0, rem_g1 - fg_rate * exec2) if task.R > 0 else 0.0
        proc['avail_q2'] = q2_start + exec2
        
        if rem_c2 <= 1e-5 and rem_g2 <= 1e-5:
            comp_time = q2_start + exec2
            total_w = w1 + w2
            total_exec = exec1 + exec2
            comp_delay = comp_time - t_arrival
            return total_exec, total_w, comp_time, comp_delay
        
        # --- Level 3 Queue (Unlimited) ---
        q3_entry = q2_start + exec2
        q3_start = max(q3_entry, proc['avail_q3'])
        w3 = q3_start - q3_entry
        
        d_c3 = rem_c2 / fc_rate
        d_g3 = rem_g2 / fg_rate if task.R > 0 else 0.0
        exec3 = max(d_c3, d_g3)
        
        proc['avail_q3'] = q3_start + exec3
        comp_time = q3_start + exec3
        total_w = w1 + w2 + w3
        total_exec = exec1 + exec2 + exec3
        comp_delay = comp_time - t_arrival
        
        return total_exec, total_w, comp_time, comp_delay


class TestMHFQSchedulerEquivalence(unittest.TestCase):

    def setUp(self):
        self.mhfq = MHFQ(num_mes=5, num_gpu_types=3)
        self.ref = EventDrivenMHFQReference(num_mes=5, num_gpu_types=3)

    def test_case1_ten_tasks_same_arrival(self):
        """Case 1: 10 tasks arriving at the same time on the same MES node."""
        tasks = [
            LRMATask(task_id=f"c1_t{i}", arrival_slot=0, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0)
            for i in range(10)
        ]
        for t in tasks:
            t.arrival_slot = 0.0
            res_code = self.mhfq.process_offloaded_task(mes_idx=0, task=t, entry_time=0.0)
            res_ref = self.ref.process_task(mes_idx=0, task=t, entry_time=0.0)
            
            # Verify exact match between production code and event-driven reference
            self.assertAlmostEqual(res_code[0], res_ref[0], delta=1e-5, msg="Processing delay must match")
            self.assertAlmostEqual(res_code[1], res_ref[1], delta=1e-5, msg="Waiting time must match")
            self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5, msg="Completion time must match")
            self.assertAlmostEqual(res_code[3], res_ref[3], delta=1e-5, msg="Completion delay must match")

    def test_case2_different_arrival_times(self):
        """Case 2: Tasks arriving at different simulation times."""
        tasks = [
            LRMATask(task_id=f"c2_t{i}", arrival_slot=i*0.5, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
            for i in range(5)
        ]
        for i, t in enumerate(tasks):
            t.arrival_slot = i * 0.5
            entry_t = i * 0.5 + 0.02  # arrival + transmission
            res_code = self.mhfq.process_offloaded_task(mes_idx=1, task=t, entry_time=entry_t)
            res_ref = self.ref.process_task(mes_idx=1, task=t, entry_time=entry_t)
            
            self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)
            self.assertAlmostEqual(res_code[3], res_ref[3], delta=1e-5)

    def test_case3_arrival_while_in_q1(self):
        """Case 3: Tasks arriving while another task is executing in Q1."""
        task_a = LRMATask(task_id="c3_ta", arrival_slot=0, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0)
        task_b = LRMATask(task_id="c3_tb", arrival_slot=0, cpu_milli=500, gpu_milli=0, gpu_spec="", duration=1.0)
        task_a.arrival_slot = 0.0
        task_b.arrival_slot = 0.05  # Arrives at t=0.05s while task_a is in Q1 (0.0 - 0.1s)
        
        res_a_code = self.mhfq.process_offloaded_task(mes_idx=2, task=task_a, entry_time=0.0)
        res_b_code = self.mhfq.process_offloaded_task(mes_idx=2, task=task_b, entry_time=0.05)
        
        res_a_ref = self.ref.process_task(mes_idx=2, task=task_a, entry_time=0.0)
        res_b_ref = self.ref.process_task(mes_idx=2, task=task_b, entry_time=0.05)
        
        self.assertAlmostEqual(res_b_code[2], res_b_ref[2], delta=1e-5)
        self.assertAlmostEqual(res_b_code[3], res_b_ref[3], delta=1e-5)

    def test_case4_arrival_while_in_q2(self):
        """Case 4: Tasks arriving while another task is executing in Q2."""
        task_a = LRMATask(task_id="c4_ta", arrival_slot=0, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0)
        task_b = LRMATask(task_id="c4_tb", arrival_slot=0, cpu_milli=500, gpu_milli=0, gpu_spec="", duration=1.0)
        task_a.arrival_slot = 0.0
        task_b.arrival_slot = 0.25  # Arrives at t=0.25s while task_a is in Q2 (0.10 - 0.40s)
        
        res_a_code = self.mhfq.process_offloaded_task(mes_idx=3, task=task_a, entry_time=0.0)
        res_b_code = self.mhfq.process_offloaded_task(mes_idx=3, task=task_b, entry_time=0.25)
        
        res_a_ref = self.ref.process_task(mes_idx=3, task=task_a, entry_time=0.0)
        res_b_ref = self.ref.process_task(mes_idx=3, task=task_b, entry_time=0.25)
        
        self.assertAlmostEqual(res_b_code[2], res_b_ref[2], delta=1e-5)
        self.assertAlmostEqual(res_b_code[3], res_b_ref[3], delta=1e-5)

    def test_case5_mixed_short_and_long_tasks(self):
        """Case 5: Interleaved short (~0.05s) and long (~1.5s) tasks."""
        t_short = LRMATask(task_id="c5_short", arrival_slot=0, cpu_milli=100, gpu_milli=0, gpu_spec="", duration=1.0)
        t_long = LRMATask(task_id="c5_long", arrival_slot=0, cpu_milli=5000, gpu_milli=0, gpu_spec="", duration=1.0)
        
        for i in range(4):
            t = t_short if i % 2 == 0 else t_long
            t.arrival_slot = 0.0
            res_code = self.mhfq.process_offloaded_task(mes_idx=0, task=t, entry_time=0.0)
            res_ref = self.ref.process_task(mes_idx=0, task=t, entry_time=0.0)
            
            self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)
            self.assertAlmostEqual(res_code[3], res_ref[3], delta=1e-5)

    def test_case6_cpu_only_tasks(self):
        """Case 6: Pure CPU-only tasks (R=0)."""
        task = LRMATask(task_id="c6_cpu", arrival_slot=0, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 0.0
        res_code = self.mhfq.process_offloaded_task(mes_idx=0, task=task, entry_time=0.0)
        res_ref = self.ref.process_task(mes_idx=0, task=task, entry_time=0.0)
        
        self.assertAlmostEqual(res_code[0], (1000*1e6)/(EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0), delta=1e-3)
        self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)

    def test_case7_gpu_tasks(self):
        """Case 7: GPU-requiring tasks (R=1)."""
        task = LRMATask(task_id="c7_gpu", arrival_slot=0, cpu_milli=10, gpu_milli=2000, gpu_spec="GPUSPEC05", duration=1.0)
        task.arrival_slot = 0.0
        res_code = self.mhfq.process_offloaded_task(mes_idx=0, task=task, entry_time=0.0)
        res_ref = self.ref.process_task(mes_idx=0, task=task, entry_time=0.0)
        
        self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)

    def test_case8_mixed_cpu_gpu_co_processing(self):
        """Case 8: Mixed CPU/GPU tasks testing D_BS = max(D_cpu, D_gpu)."""
        task = LRMATask(task_id="c8_mixed", arrival_slot=0, cpu_milli=1000, gpu_milli=4000, gpu_spec="GPUSPEC20", duration=1.0)
        task.arrival_slot = 0.0
        res_code = self.mhfq.process_offloaded_task(mes_idx=0, task=task, entry_time=0.0)
        res_ref = self.ref.process_task(mes_idx=0, task=task, entry_time=0.0)
        
        d_c = (1000 * 1e6) / (EnvConfig.MES_TOTAL_CPU_CAPACITY/3.0)
        d_g = (4000 * 1e6) / EnvConfig.MES_GPU_CAPACITY
        expected_d = max(d_c, d_g)
        self.assertAlmostEqual(res_code[0], expected_d, delta=1e-3)
        self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)

    def test_case9_multiple_gpu_resource_types(self):
        """Case 9: Multiple GPU resource types (R=0, R=1, R=2, R=3) on same MES."""
        t_r0 = LRMATask(task_id="c9_r0", arrival_slot=0, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        t_r1 = LRMATask(task_id="c9_r1", arrival_slot=0, cpu_milli=1000, gpu_milli=1000, gpu_spec="GPUSPEC05", duration=1.0)
        t_r2 = LRMATask(task_id="c9_r2", arrival_slot=0, cpu_milli=1000, gpu_milli=1000, gpu_spec="GPUSPEC20", duration=1.0)
        t_r3 = LRMATask(task_id="c9_r3", arrival_slot=0, cpu_milli=1000, gpu_milli=1000, gpu_spec="GPUSPEC33", duration=1.0)
        
        for t in [t_r0, t_r1, t_r2, t_r3]:
            t.arrival_slot = 0.0
            res_code = self.mhfq.process_offloaded_task(mes_idx=2, task=t, entry_time=0.0)
            res_ref = self.ref.process_task(mes_idx=2, task=t, entry_time=0.0)
            self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)

    def test_case10_multiple_mes_nodes(self):
        """Case 10: Tasks offloaded to multiple independent MES nodes (j=0..4)."""
        for j in range(5):
            t = LRMATask(task_id=f"c10_mes{j}", arrival_slot=0, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
            t.arrival_slot = 0.0
            res_code = self.mhfq.process_offloaded_task(mes_idx=j, task=t, entry_time=0.0)
            res_ref = self.ref.process_task(mes_idx=j, task=t, entry_time=0.0)
            self.assertAlmostEqual(res_code[2], res_ref[2], delta=1e-5)

if __name__ == "__main__":
    unittest.main()
