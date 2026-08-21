import unittest
import numpy as np
from src.config import EnvConfig
from src.data_loader import LRMATask
from src.lyapunov import MHFQProcessor, MHFQ, FCFSQueue, MMCQueue, QueuedTaskState
from src.wireless import WirelessModel

class TestExecutionUnits(unittest.TestCase):

    def test_cpu_only_local_execution_units(self):
        """1. Verify CPU-only local execution processing duration and units."""
        # 1000 milli-CPU = 10^9 cycles. Local CPU capacity = 2.0 GHz (2*10^9 cycles/s).
        # Expected D_cpu = 10^9 / 2*10^9 = 0.5 seconds.
        task = LRMATask(task_id="cpu_only_local", arrival_slot=0, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        c_time = (task.C * 1e6) / float(EnvConfig.LOCAL_CPU_CAPACITY)
        g_time = 0.0
        d_ed = max(c_time, g_time)
        
        self.assertAlmostEqual(c_time, 0.5, delta=1e-5, msg="1000 milli-CPU on 2.0GHz CPU must take 0.5s")
        self.assertEqual(d_ed, 0.5)

    def test_gpu_only_local_execution_units(self):
        """2. Verify GPU-only local execution processing duration and units."""
        # 2000 milli-GPU = 2*10^9 cycles. Local GPU capacity = 4.0 GHz.
        # Expected D_gpu = 2*10^9 / 4*10^9 = 0.5 seconds.
        task = LRMATask(task_id="gpu_only_local", arrival_slot=0, cpu_milli=0, gpu_milli=2000, gpu_spec="GPUSPEC10", duration=1.0)
        c_time = (task.C * 1e6) / float(EnvConfig.LOCAL_CPU_CAPACITY)
        g_time = (task.G * 1e6) / float(EnvConfig.LOCAL_GPU_CAPACITY)
        d_ed = max(c_time, g_time)
        
        self.assertAlmostEqual(g_time, 0.5, delta=1e-5, msg="2000 milli-GPU on 4.0GHz GPU must take 0.5s")
        self.assertEqual(d_ed, 0.5)

    def test_cpu_gpu_co_processing_bottleneck(self):
        """3 & 4. Verify CPU+GPU co-processing bottleneck D_local = max(D_cpu, D_gpu)."""
        # CPU: 1000 milli-CPU -> 0.5s on 2.0GHz. GPU: 4000 milli-GPU -> 1.0s on 4.0GHz.
        # Expected D_local = max(0.5, 1.0) = 1.0s.
        task = LRMATask(task_id="co_proc", arrival_slot=0, cpu_milli=1000, gpu_milli=4000, gpu_spec="GPUSPEC20", duration=1.0)
        c_time = (task.C * 1e6) / float(EnvConfig.LOCAL_CPU_CAPACITY)
        g_time = (task.G * 1e6) / float(EnvConfig.LOCAL_GPU_CAPACITY)
        d_ed = max(c_time, g_time)
        
        self.assertAlmostEqual(c_time, 0.5, delta=1e-5)
        self.assertAlmostEqual(g_time, 1.0, delta=1e-5)
        self.assertEqual(d_ed, 1.0, "Co-processing delay must equal max(D_cpu, D_gpu)")

    def test_local_accounting_identity_completion_time_minus_arrival(self):
        """5 & 6. Verify completion_delay = completion_time - arrival_time = waiting + processing."""
        arrival_time = 10.0
        ed_avail_time = 15.0  # Server free at t=15.0
        d_ed = 0.5
        
        top_local = max(arrival_time, ed_avail_time)
        w_ed = top_local - arrival_time  # 5.0s
        completion_time = top_local + d_ed  # 15.5s
        completion_delay = completion_time - arrival_time  # 5.5s
        
        self.assertAlmostEqual(completion_delay, 5.5, delta=1e-5)
        self.assertAlmostEqual(w_ed + d_ed, completion_delay, delta=1e-5, msg="waiting + processing must equal completion_delay")

    def test_offloaded_transmission_plus_mhfq_delay_accounting(self):
        """7 & 8. Verify offloaded transmission + waiting + processing accounting & no double counting."""
        wireless = WirelessModel()
        t_arrival = 10.0
        t_trans = 0.05
        to_entry = t_arrival + t_trans  # 10.05s
        
        proc = MHFQProcessor(mes_idx=0, gpu_type=0)
        proc.avail_q1 = 12.0  # Queue available at t=12.0
        
        task = LRMATask(task_id="offload_acc", arrival_slot=10, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 10.0
        
        # MES CPU: 3.33 GHz -> D_bs = 10^9 / 3.33*10^9 = 0.3s
        d_bs, w_bs, completion_time, completion_delay = proc.process_task(task, entry_time=to_entry, f_c=3.33e9, f_g=8e9)
        
        # Expected:
        # to_entry = 10.05
        # service_start = max(10.05, 12.0) = 12.0
        # w_bs = 12.0 - 10.05 = 1.95s (waiting at MES after transmission)
        # completion_time = 12.0 + 0.3 = 12.3s
        # completion_delay = 12.3 - 10.0 = 2.3s
        self.assertAlmostEqual(w_bs, 1.95, delta=0.01)
        self.assertAlmostEqual(completion_time, 12.3, delta=0.01)
        self.assertAlmostEqual(completion_delay, 2.3, delta=0.01)
        
        # Verify identity: transmission_delay + waiting_time + processing_delay = completion_delay
        sum_delays = t_trans + w_bs + d_bs
        self.assertAlmostEqual(sum_delays, completion_delay, delta=1e-5, msg="t_trans + w_bs + d_bs must equal completion_delay without double counting")

    def test_mes_cpu_execution_units(self):
        """9. Verify MES CPU execution units."""
        # 1000 milli-CPU = 10^9 cycles. MES CPU capacity = 10.0 GHz total, split 3 ways = 3.333 GHz.
        # Expected D_c = 10^9 / 3.333*10^9 = 0.30 seconds.
        c_req = 1000.0 * 1e6
        f_c = EnvConfig.MES_TOTAL_CPU_CAPACITY / 3.0
        d_c = c_req / f_c
        self.assertAlmostEqual(d_c, 0.30, delta=0.01)

    def test_mes_gpu_execution_units(self):
        """10. Verify MES GPU execution units."""
        # 2000 milli-GPU = 2*10^9 cycles. MES GPU capacity = 8.0 GHz.
        # Expected D_g = 2*10^9 / 8.0*10^9 = 0.25 seconds.
        g_req = 2000.0 * 1e6
        f_g = EnvConfig.MES_GPU_CAPACITY
        d_g = g_req / f_g
        self.assertAlmostEqual(d_g, 0.25, delta=0.01)

    def test_queue_backlog_decay_unit_consistency(self):
        """11. Verify queue backlog decay formula unit consistency (both sides in bits)."""
        # MES_TOTAL_CPU_CAPACITY = 10^10 cycles/s, TAU = 1.0 s, RHO = 10.0 cycles/bit.
        # Processed capacity = 10^10 * 1.0 / 10.0 = 10^9 bits = 1 Gb.
        processed_cap_mes = (EnvConfig.MES_TOTAL_CPU_CAPACITY * EnvConfig.TAU) / float(EnvConfig.RHO)
        self.assertAlmostEqual(processed_cap_mes, 1e9, delta=1e-5, msg="MES decay capacity must equal 10^9 bits")
        
        processed_cap_ed = (EnvConfig.LOCAL_CPU_CAPACITY * EnvConfig.TAU) / float(EnvConfig.RHO)
        self.assertAlmostEqual(processed_cap_ed, 2e8, delta=1e-5, msg="ED decay capacity must equal 2*10^8 bits")

    def test_mhfq_service_time_consistency(self):
        """12. Verify MHFQ service-time progression across time slices Q1(0.1s), Q2(0.3s), Q3."""
        proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        
        # Small task (~0.05s) completes in Q1
        task_small = LRMATask(task_id="tsmall", arrival_slot=0, cpu_milli=166, gpu_milli=0, gpu_spec="", duration=1.0)
        d1, w1, ctime1, cdelay1 = proc.process_task(task_small, entry_time=0.0, f_c=3.33e9, f_g=8e9)
        self.assertTrue(d1 <= 0.1, "Small task must finish within Q1 slice 0.1s")
        
        # Large task (~0.5s) completes after migrating Q1(0.1s) -> Q2(0.3s) -> Q3(0.1s)
        proc2 = MHFQProcessor(mes_idx=0, gpu_type=1)
        task_large = LRMATask(task_id="tlarge", arrival_slot=0, cpu_milli=1665, gpu_milli=0, gpu_spec="", duration=1.0)
        d2, w2, ctime2, cdelay2 = proc2.process_task(task_large, entry_time=0.0, f_c=3.33e9, f_g=8e9)
        self.assertAlmostEqual(d2, 0.5, delta=0.05, msg="Large task total service time must accumulate across slices to ~0.5s")

if __name__ == "__main__":
    unittest.main()
