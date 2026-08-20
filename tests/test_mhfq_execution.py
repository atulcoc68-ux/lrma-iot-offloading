import unittest
import numpy as np
from collections import deque
from src.config import EnvConfig
from src.data_loader import LRMATask
from src.lyapunov import MHFQProcessor, MHFQ, FCFSQueue, MMCQueue, QueuedTaskState

class TestMHFQExecution(unittest.TestCase):

    def test_delay_accounting_regression_arrival_subtraction(self):
        """
        Regression test: Verify arrival time subtraction for completion_delay.
        Example: arrival = 20.0, service_start = 1200.0, d_bs = 0.5
        expected: completion_time = 1200.5, completion_delay = 1180.5.
        Server availability timeline must remain on absolute timeline (1200.1s in Q1), NOT elapsed delay (1180.5s).
        """
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        proc.avail_q1 = 1200.0  # Simulate prior backlog advancing server availability to 1200.0s
        
        # Task arrives at slot t = 20.0
        task1 = LRMATask(task_id="task_slot20", arrival_slot=20, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0) # ~0.5s compute
        task1.arrival_slot = 20.0
        
        # Process task arriving at entry_time = 20.05 (slot 20 + transmission 0.05s)
        d_bs, w_bs, comp_time, comp_delay = proc.process_task(task1, entry_time=20.05, f_c=3.33e9, f_g=8e9)
        
        # Expected values:
        # service_start_q1 = max(20.05, 1200.0) = 1200.0
        # d_bs = 0.5
        # comp_time = 1200.5
        # comp_delay = 1200.5 - 20.0 = 1180.5
        self.assertAlmostEqual(comp_time, 1200.5, delta=0.05, msg="Completion time must be absolute timestamp ~1200.5s")
        self.assertAlmostEqual(comp_delay, 1180.5, delta=0.05, msg="Completion delay must subtract task arrival time ~1180.5s")
        
        # Verify next task's server availability timeline in Q1 remains on absolute simulation timeline (1200.1s after 0.1s Q1 slice), NOT 1180.5s
        task2 = LRMATask(task_id="task_next", arrival_slot=21, cpu_milli=10, gpu_milli=0, gpu_spec="", duration=1.0)
        d_bs2, w_bs2, comp_time2, comp_delay2 = proc.process_task(task2, entry_time=21.05, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(proc.avail_q1, 1200.1 + d_bs2, delta=0.05, msg="Server availability in Q1 must remain on absolute timeline ~1200.1s")

    def test_fcfs_delay_accounting(self):
        """Regression test: FCFS returns completion_time and completion_delay correctly."""
        fcfs = FCFSQueue(num_mes=5)
        fcfs.avail_time[0] = 1200.0
        
        task1 = LRMATask(task_id="fcfs_task", arrival_slot=20, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0)
        task1.arrival_slot = 20.0
        
        d_bs, w_bs, comp_time, comp_delay = fcfs.process_offloaded_task(mes_idx=0, task=task1, entry_time=20.05, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(comp_time, 1200.5, delta=0.05)
        self.assertAlmostEqual(comp_delay, 1180.5, delta=0.05)
        self.assertEqual(fcfs.avail_time[0], comp_time, "FCFS server timeline must equal absolute completion time")

    def test_mmc_delay_accounting(self):
        """Regression test: MMC returns completion_time and completion_delay correctly."""
        mmc = MMCQueue(num_mes=5, c_channels=2)
        mmc.channel_avail[0] = [1200.0, 1200.0]
        
        task1 = LRMATask(task_id="mmc_task", arrival_slot=20, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0)
        task1.arrival_slot = 20.0
        
        d_bs, w_bs, comp_time, comp_delay = mmc.process_offloaded_task(mes_idx=0, task=task1, entry_time=20.05, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(comp_time, 1200.5, delta=0.05)
        self.assertAlmostEqual(comp_delay, 1180.5, delta=0.05)
        self.assertEqual(mmc.channel_avail[0][0], comp_time, "MMC channel timeline must equal absolute completion time")

    def test_task_enters_q1(self):
        """1: Task enters Q1 container queue."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        task = LRMATask(task_id="task_q1_init", arrival_slot=1, cpu_milli=10, gpu_milli=0, gpu_spec="", duration=1.0)
        
        state = QueuedTaskState(task, entry_time=1.0, ed_id=3)
        self.assertEqual(state.queue_level, 1)
        self.assertEqual(state.ed_id, 3)
        self.assertAlmostEqual(state.remaining_cpu_cycles, 10 * 1e6)
        self.assertAlmostEqual(state.remaining_gpu_cycles, 0.0)

    def test_q1_completion_small_task(self):
        """Small task completes in Q1."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        task = LRMATask(task_id="small_task_1", arrival_slot=1, cpu_milli=10, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0
        d_bs, w_bs, comp_time, comp_delay = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(w_bs, 0.0, delta=1e-5)
        self.assertTrue(d_bs <= 0.1)
        self.assertAlmostEqual(comp_time, 1.0 + d_bs, delta=1e-5)
        self.assertAlmostEqual(comp_delay, d_bs, delta=1e-5)

    def test_q1_to_q2_migration(self):
        """Q1 -> Q2 migration: Medium task exceeding Q1 service moves to Q2 and preserves remaining work."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        task = LRMATask(task_id="med_task_1", arrival_slot=1, cpu_milli=666, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0
        d_bs, w_bs, comp_time, comp_delay = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(d_bs, 0.2, delta=0.02)
        self.assertTrue(proc.avail_q1 > 1.0)
        self.assertTrue(proc.avail_q2 > 1.0)

    def test_q2_to_q3_migration(self):
        """Q2 -> Q3 migration & Q3 completion: Large task exceeding Q2 service moves to Q3."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        task = LRMATask(task_id="large_task_1", arrival_slot=1, cpu_milli=2664, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0
        d_bs, w_bs, comp_time, comp_delay = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(d_bs, 0.8, delta=0.05)
        self.assertTrue(proc.avail_q3 > 1.0)

    def test_cpu_only_task_no_gpu(self):
        """CPU-only task has no GPU execution component."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=0)
        task = LRMATask(task_id="cpu_only_1", arrival_slot=1, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0
        self.assertEqual(task.R, 0)
        d_bs, w_bs, comp_time, comp_delay = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        expected_d = (1000 * 1e6) / 3.33e9
        self.assertAlmostEqual(d_bs, expected_d, delta=1e-3)

    def test_heterogeneous_cpu_gpu_task(self):
        """Heterogeneous CPU+GPU task uses max(D_c, D_g) co-processing bottleneck."""
        mhfq = MHFQ(num_mes=5, num_gpu_types=3)
        task = LRMATask(task_id="gpu_task_r2", arrival_slot=1, cpu_milli=100, gpu_milli=500, gpu_spec="GPUSPEC20", duration=1.0)
        task.arrival_slot = 1.0
        self.assertEqual(task.R, 2)
        
        d_bs, w_bs, comp_time, comp_delay = mhfq.process_offloaded_task(mes_idx=1, task=task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        d_c = (100 * 1e6) / 3.33e9
        d_g = (500 * 1e6) / 8e9
        expected_d = max(d_c, d_g)
        self.assertAlmostEqual(d_bs, expected_d, delta=1e-3)

    def test_fifo_ordering_and_waiting_time(self):
        """FIFO ordering and waiting time with multiple tasks."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        task1 = LRMATask(task_id="task_front", arrival_slot=1, cpu_milli=666, gpu_milli=0, gpu_spec="", duration=1.0)
        task2 = LRMATask(task_id="task_behind", arrival_slot=1, cpu_milli=666, gpu_milli=0, gpu_spec="", duration=1.0)
        task1.arrival_slot = 1.0
        task2.arrival_slot = 1.0
        
        res1 = proc.process_task(task1, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        res2 = proc.process_task(task2, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(res1[1], 0.0, delta=1e-5)
        self.assertAlmostEqual(res2[1], 0.1, delta=0.02)

    def test_mhfq_determinism(self):
        """MHFQ is deterministic for fixed workload and seed."""
        mhfq1 = MHFQ(num_mes=5, num_gpu_types=3)
        mhfq2 = MHFQ(num_mes=5, num_gpu_types=3)
        task = LRMATask(task_id="det_task", arrival_slot=1, cpu_milli=500, gpu_milli=100, gpu_spec="GPUSPEC10", duration=1.0)
        task.arrival_slot = 1.0
        
        res1 = mhfq1.process_offloaded_task(mes_idx=2, task=task, entry_time=5.0)
        res2 = mhfq2.process_offloaded_task(mes_idx=2, task=task, entry_time=5.0)
        
        self.assertEqual(res1, res2)

if __name__ == "__main__":
    unittest.main()
