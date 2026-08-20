import unittest
import numpy as np
from collections import deque
from src.config import EnvConfig
from src.data_loader import LRMATask
from src.lyapunov import MHFQProcessor, MHFQ, FCFSQueue, MMCQueue, QueuedTaskState

class TestMHFQExecution(unittest.TestCase):

    def test_task_enters_q1(self):
        """1: Task enters Q1 container queue."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        task = LRMATask(task_id="task_q1_init", arrival_slot=1, cpu_milli=10, gpu_milli=0, gpu_spec="", duration=1.0)
        
        # Verify initial QueuedTaskState attributes
        state = QueuedTaskState(task, entry_time=1.0, ed_id=3)
        self.assertEqual(state.queue_level, 1)
        self.assertEqual(state.ed_id, 3)
        self.assertAlmostEqual(state.remaining_cpu_cycles, 10 * 1e6)
        self.assertAlmostEqual(state.remaining_gpu_cycles, 0.0)

    def test_q1_completion_small_task(self):
        """Small task completes in Q1."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        # Small task: C=10 milli-CPU -> 10 * 1e6 = 1e7 cycles -> 1e7 / 3.33e9 = 0.003s < 0.1s (tau_1^ves)
        task = LRMATask(task_id="small_task_1", arrival_slot=1, cpu_milli=10, gpu_milli=0, gpu_spec="", duration=1.0)
        d_bs, w_bs, completion_time = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(w_bs, 0.0, delta=1e-5, msg="Empty queue should have 0 wait time")
        self.assertTrue(d_bs <= 0.1, f"Small task execution delay {d_bs} should fit in Q1 (0.1s)")
        self.assertAlmostEqual(completion_time, 1.0 + d_bs, delta=1e-5)

    def test_q1_to_q2_migration(self):
        """Q1 -> Q2 migration: Medium task exceeding Q1 service moves to Q2 and preserves remaining work."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        # Medium task requiring 0.2s CPU time: 0.2s * 3.33e9 = 6.66e8 cycles -> 666 milli-CPU
        task = LRMATask(task_id="med_task_1", arrival_slot=1, cpu_milli=666, gpu_milli=0, gpu_spec="", duration=1.0)
        d_bs, w_bs, completion_time = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(d_bs, 0.2, delta=0.02, msg="Medium task total delay should equal required compute time ~0.2s")
        self.assertTrue(proc.avail_q1 > 1.0, "Q1 availability timeline must advance")
        self.assertTrue(proc.avail_q2 > 1.0, "Q2 availability timeline must advance on migration")

    def test_q2_to_q3_migration(self):
        """Q2 -> Q3 migration & Q3 completion: Large task exceeding Q2 service moves to Q3."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        # Large task requiring 0.8s compute time: 0.8s * 3.33e9 = 2.664e9 cycles -> 2664 milli-CPU
        task = LRMATask(task_id="large_task_1", arrival_slot=1, cpu_milli=2664, gpu_milli=0, gpu_spec="", duration=1.0)
        d_bs, w_bs, completion_time = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(d_bs, 0.8, delta=0.05, msg="Large task total delay should equal required compute time ~0.8s")
        self.assertTrue(proc.avail_q3 > 1.0, "Q3 availability timeline must advance on migration to Q3")

    def test_cpu_only_task_no_gpu(self):
        """CPU-only task has no GPU execution component."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=0)
        task = LRMATask(task_id="cpu_only_1", arrival_slot=1, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        self.assertEqual(task.R, 0)
        d_bs, w_bs, comp = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        expected_d = (1000 * 1e6) / 3.33e9
        self.assertAlmostEqual(d_bs, expected_d, delta=1e-3)

    def test_heterogeneous_cpu_gpu_task(self):
        """Heterogeneous CPU+GPU task uses max(D_c, D_g) co-processing bottleneck."""
        mhfq = MHFQ(num_mes=5, num_gpu_types=3)
        # Task with GPU requirement R=2
        task = LRMATask(task_id="gpu_task_r2", arrival_slot=1, cpu_milli=100, gpu_milli=500, gpu_spec="GPUSPEC20", duration=1.0)
        self.assertEqual(task.R, 2)
        
        d_bs, w_bs, comp = mhfq.process_offloaded_task(mes_idx=1, task=task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        d_c = (100 * 1e6) / 3.33e9
        d_g = (500 * 1e6) / 8e9
        expected_d = max(d_c, d_g)
        self.assertAlmostEqual(d_bs, expected_d, delta=1e-3)

    def test_fifo_ordering_and_waiting_time(self):
        """FIFO ordering and waiting time with multiple tasks."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        # Task 1 requiring 0.2s CPU time -> Spends tau_1^ves = 0.1s in Q1 then migrates to Q2
        task1 = LRMATask(task_id="task_front", arrival_slot=1, cpu_milli=666, gpu_milli=0, gpu_spec="", duration=1.0)
        task2 = LRMATask(task_id="task_behind", arrival_slot=1, cpu_milli=666, gpu_milli=0, gpu_spec="", duration=1.0)
        
        d1, w1, comp1 = proc.process_task(task1, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        d2, w2, comp2 = proc.process_task(task2, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(w1, 0.0, delta=1e-5)
        # Task 2 in Q1 waits for Task 1's Q1 time slice (0.1s)
        self.assertAlmostEqual(w2, 0.1, delta=0.02, msg="Task 2 wait in Q1 should equal Task 1's Q1 time slice (0.1s)")

    def test_multiple_mess_and_gpu_types(self):
        """Multiple MESs and GPU resource types."""
        mhfq = MHFQ(num_mes=5, num_gpu_types=3)
        self.assertEqual(len(mhfq.processors), 5)
        for j in range(5):
            self.assertEqual(len(mhfq.processors[j]), 4)  # Types 0, 1, 2, 3

        task_mes0_r1 = LRMATask(task_id="m0r1", arrival_slot=1, cpu_milli=100, gpu_milli=100, gpu_spec="GPUSPEC05", duration=1.0)
        task_mes3_r3 = LRMATask(task_id="m3r3", arrival_slot=1, cpu_milli=200, gpu_milli=200, gpu_spec="GPUSPEC33", duration=1.0)
        
        res1 = mhfq.process_offloaded_task(mes_idx=0, task=task_mes0_r1, entry_time=1.0)
        res2 = mhfq.process_offloaded_task(mes_idx=3, task=task_mes3_r3, entry_time=1.0)
        
        self.assertEqual(task_mes0_r1.R, 1)
        self.assertEqual(task_mes3_r3.R, 3)
        self.assertTrue(res1[0] > 0)
        self.assertTrue(res2[0] > 0)

    def test_fcfs_fifo_queue(self):
        """FCFS maintains actual FIFO queue per MES."""
        fcfs = FCFSQueue(num_mes=5)
        task1 = LRMATask(task_id="fcfs_1", arrival_slot=1, cpu_milli=333, gpu_milli=0, gpu_spec="", duration=1.0)
        task2 = LRMATask(task_id="fcfs_2", arrival_slot=1, cpu_milli=333, gpu_milli=0, gpu_spec="", duration=1.0)
        
        d1, w1, comp1 = fcfs.process_offloaded_task(mes_idx=0, task=task1, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        d2, w2, comp2 = fcfs.process_offloaded_task(mes_idx=0, task=task2, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        self.assertAlmostEqual(comp1, 1.0 + d1, delta=1e-4)
        self.assertAlmostEqual(comp2, comp1 + d2, delta=1e-4)

    def test_mmc_parallel_channel_queues(self):
        """M/M/C maintains actual channel queues and allows C simultaneous services."""
        mmc = MMCQueue(num_mes=5, c_channels=2)
        task1 = LRMATask(task_id="mmc_1", arrival_slot=1, cpu_milli=333, gpu_milli=0, gpu_spec="", duration=1.0)
        task2 = LRMATask(task_id="mmc_2", arrival_slot=1, cpu_milli=333, gpu_milli=0, gpu_spec="", duration=1.0)
        
        # Both tasks arrive at entry_time=1.0
        d1, w1, comp1 = mmc.process_offloaded_task(mes_idx=0, task=task1, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        d2, w2, comp2 = mmc.process_offloaded_task(mes_idx=0, task=task2, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        
        # Because C=2, both tasks are served simultaneously on separate channels -> w2 should be 0!
        self.assertAlmostEqual(w1, 0.0, delta=1e-5)
        self.assertAlmostEqual(w2, 0.0, delta=1e-5, msg="Second task on 2-channel M/M/C should have 0 wait time")

    def test_mhfq_determinism(self):
        """MHFQ is deterministic for fixed workload and seed."""
        mhfq1 = MHFQ(num_mes=5, num_gpu_types=3)
        mhfq2 = MHFQ(num_mes=5, num_gpu_types=3)
        task = LRMATask(task_id="det_task", arrival_slot=1, cpu_milli=500, gpu_milli=100, gpu_spec="GPUSPEC10", duration=1.0)
        
        res1 = mhfq1.process_offloaded_task(mes_idx=2, task=task, entry_time=5.0)
        res2 = mhfq2.process_offloaded_task(mes_idx=2, task=task, entry_time=5.0)
        
        self.assertEqual(res1, res2)

if __name__ == "__main__":
    unittest.main()
