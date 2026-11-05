import unittest
import torch
import numpy as np
from collections import deque

from src.config import EnvConfig
from src.data_loader import AlibabaWorkloadLoader, LRMATask
from src.lstm_model import WorkloadPredictor
from src.lyapunov import MHFQProcessor, MHFQ, AdaptiveMHFQProcessor, AdaptiveMHFQ, QueuedTaskState
from src.environment import LRMA_Environment
from evaluate import evaluate_policy


class TestAdaptiveMHFQ(unittest.TestCase):
    """
    Comprehensive test suite for Adaptive MHFQ (Self-Adaptive Multi-Level Heterogeneous Feedback Queue).
    Validates dynamic quantum adaptation, near-completion slack, anti-starvation aging,
    heterogeneous CPU/GPU co-processing, environment integration, and original LSTM model enforcement.
    """

    def test_adaptive_mhfq_processor_initialization(self):
        """Verifies processor default configuration and telemetry tracking."""
        proc = AdaptiveMHFQProcessor(mes_idx=0, gpu_type=1)
        self.assertEqual(proc.base_time_slices, [0.1, 0.3, 0.6])
        self.assertEqual(proc.slack_ratio, 0.15)
        self.assertEqual(proc.aging_threshold, 1.0)
        self.assertEqual(proc.tasks_processed, 0)
        self.assertEqual(proc.slack_completions, 0)
        self.assertEqual(proc.priority_boosts, 0)

    def test_dynamic_time_slices_scaling(self):
        """Verifies time slices adaptively scale under backlog pressure."""
        proc = AdaptiveMHFQProcessor(mes_idx=0, gpu_type=1, ref_backlog=1e7)
        slices_empty = proc.get_adaptive_time_slices()
        self.assertAlmostEqual(slices_empty[0], 0.1, places=3)
        self.assertAlmostEqual(slices_empty[1], 0.3, places=3)
        self.assertAlmostEqual(slices_empty[2], 0.6, places=3)

        # Inject artificial backlog into Q1
        fake_task = LRMATask(task_id="heavy_backlog", arrival_slot=1, cpu_milli=1000, gpu_milli=0, gpu_spec="", duration=1.0)
        fake_task.size = 1e7  # 10 Mbits backlog -> load_ratio = tanh(1.0) ~ 0.7616
        proc.q1.append(QueuedTaskState(fake_task, entry_time=1.0))

        slices_loaded = proc.get_adaptive_time_slices()
        self.assertGreater(slices_loaded[0], 0.1)
        self.assertGreater(slices_loaded[1], 0.3)
        self.assertGreater(slices_loaded[2], 0.6)
        # Expected tau1 = 0.1 * (1 + 0.20 * 0.7616) ~ 0.1152s
        expected_tau1 = 0.1 * (1.0 + 0.20 * np.tanh(1.0))
        self.assertAlmostEqual(slices_loaded[0], expected_tau1, places=3)

    def test_near_completion_slack_in_q1(self):
        """
        Near-completion slack: A task requiring 0.11s of computation (within 15% slack of 0.10s)
        finishes directly in Q1 under Adaptive MHFQ, whereas static MHFQ forces a costly migration to Q2.
        """
        # 1. Static MHFQ Processor
        static_proc = MHFQProcessor(mes_idx=0, gpu_type=1)
        # Task with 0.11s execution: c_cycles = 0.11 * 3.33e9 ~ 366.3e6
        cpu_milli = 366.3
        task_static = LRMATask(task_id="task_slack", arrival_slot=1, cpu_milli=cpu_milli, gpu_milli=0, gpu_spec="", duration=1.0)
        task_static.arrival_slot = 1.0

        d_s, w_s, c_s, delay_s = static_proc.process_task(task_static, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        # In static MHFQ: runs 0.10s in Q1, migrates to Q2, runs 0.01s in Q2
        self.assertTrue(static_proc.avail_q2 > 1.0, "Static MHFQ must have migrated task to Q2")

        # 2. Adaptive MHFQ Processor
        adapt_proc = AdaptiveMHFQProcessor(mes_idx=0, gpu_type=1, slack_ratio=0.15)
        task_adapt = LRMATask(task_id="task_slack", arrival_slot=1, cpu_milli=cpu_milli, gpu_milli=0, gpu_spec="", duration=1.0)
        task_adapt.arrival_slot = 1.0

        d_a, w_a, c_a, delay_a = adapt_proc.process_task(task_adapt, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        # Under Adaptive MHFQ: completed directly in Q1 via slack!
        self.assertEqual(adapt_proc.q1_completions, 1, "Task should complete in Q1 via slack")
        self.assertEqual(adapt_proc.q2_completions, 0, "Task should NOT have migrated to Q2")
        self.assertEqual(adapt_proc.slack_completions, 1, "Slack completion counter must be 1")
        self.assertEqual(adapt_proc.avail_q2, 0.0, "Q2 server timeline should not be touched")
        self.assertAlmostEqual(d_a, 0.11, delta=0.01)

    def test_task_exceeding_slack_demotes_to_q2(self):
        """Tasks exceeding the slack threshold (e.g. 0.25s compute) still cleanly migrate to Q2."""
        adapt_proc = AdaptiveMHFQProcessor(mes_idx=0, gpu_type=1, slack_ratio=0.15)
        # Compute time = 0.25s -> c_cycles = 0.25 * 3.33e9 = 832.5e6 -> cpu_milli = 832.5
        task = LRMATask(task_id="med_task", arrival_slot=1, cpu_milli=832.5, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0

        d, w, comp, delay = adapt_proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        self.assertEqual(adapt_proc.q1_completions, 0)
        self.assertEqual(adapt_proc.q2_completions, 1)
        self.assertTrue(adapt_proc.avail_q2 > 1.0)

    def test_anti_starvation_priority_boost(self):
        """
        Anti-starvation: When a task suffers accumulated waiting time exceeding aging_threshold,
        it receives an adaptive priority boost.
        """
        proc = AdaptiveMHFQProcessor(mes_idx=0, gpu_type=1, aging_threshold=1.0)
        # Advance Q1 server availability so task experiences wait in Q1
        proc.avail_q1 = 2.5
        task = LRMATask(task_id="starved_task", arrival_slot=1, cpu_milli=832.5, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0

        # Enters at t=1.0, waits in Q1 until 2.5s (wait = 1.5s > aging_threshold 1.0s)
        d, w, comp, delay = proc.process_task(task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        self.assertGreaterEqual(proc.priority_boosts, 1, "Priority boost must trigger when wait >= aging_threshold")

    def test_heterogeneous_cpu_gpu_co_processing(self):
        """Validates co-processing bottleneck max(D_c, D_g) with GPU acceleration under Adaptive MHFQ."""
        mhfq = AdaptiveMHFQ(num_mes=5, num_gpu_types=3)
        # CPU compute: 100M cycles / 3.33e9 = 0.030s
        # GPU compute: 800M cycles / 8.0e9 = 0.100s -> Bottleneck is GPU: 0.100s
        task = LRMATask(task_id="gpu_co_proc", arrival_slot=1, cpu_milli=100, gpu_milli=800, gpu_spec="GPUSPEC20", duration=1.0)
        task.arrival_slot = 1.0
        self.assertEqual(task.R, 2)

        d_bs, w_bs, comp_time, comp_delay = mhfq.process_offloaded_task(mes_idx=2, task=task, entry_time=1.0, f_c=3.33e9, f_g=8e9)
        self.assertAlmostEqual(d_bs, 0.10, places=2)

    def test_adaptive_stats_aggregation(self):
        """Verifies get_adaptive_stats() aggregates telemetry across all nodes and processors."""
        mhfq = AdaptiveMHFQ(num_mes=5, num_gpu_types=3)
        task = LRMATask(task_id="t_stats", arrival_slot=1, cpu_milli=366.3, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0

        mhfq.process_offloaded_task(mes_idx=0, task=task, entry_time=1.0)
        stats = mhfq.get_adaptive_stats()
        self.assertEqual(stats['total_tasks'], 1)
        self.assertIn('slack_completions', stats)
        self.assertIn('priority_boosts', stats)
        self.assertIn('slack_completion_rate', stats)

    def test_environment_integration_with_adaptive_mhfq(self):
        """Verifies LRMA_Environment seamlessly instantiates and runs with queue_type='AdaptiveMHFQ'."""
        loader = AlibabaWorkloadLoader()
        predictor = WorkloadPredictor(input_dim=4, hidden_dim=64, output_dim=4, num_layers=2)
        env = LRMA_Environment(loader, predictor, EnvConfig, queue_type='AdaptiveMHFQ', num_ed=5)

        self.assertIsInstance(env.mhfq, AdaptiveMHFQ)
        env.update_time_slot(1, {})
        task = LRMATask(task_id="env_task", arrival_slot=1, cpu_milli=100, gpu_milli=0, gpu_spec="", duration=1.0)
        task.arrival_slot = 1.0

        res = env.step_task_offloading(ed_idx=0, task=task, ed_action=1, cloud_action=2)
        self.assertTrue(res['is_offloaded'])
        self.assertEqual(res['mes_assigned'], 2)
        self.assertGreater(res['delay'], 0.0)

    def test_original_unquantized_lstm_predictor(self):
        """
        Strict verification: Original full-precision (FP32) PyTorch LSTM model is used,
        without any INT8 quantization wrappers.
        """
        predictor = WorkloadPredictor(input_dim=4, hidden_dim=64, output_dim=4, num_layers=2)
        self.assertIsInstance(predictor.lstm, torch.nn.LSTM)
        self.assertIsInstance(predictor.fc, torch.nn.Linear)
        # Check weights are float32
        self.assertEqual(predictor.lstm.weight_ih_l0.dtype, torch.float32)
        self.assertEqual(predictor.fc.weight.dtype, torch.float32)

    def test_evaluate_policy_with_adaptive_mhfq(self):
        """End-to-end evaluation with AdaptiveMHFQ over 5 slots."""
        summary, ed_q, mes_q = evaluate_policy(
            algorithm='LRMA', num_ed=5, queue_type='AdaptiveMHFQ',
            seed=42, total_slots=5
        )
        self.assertEqual(summary['queue_type'], 'AdaptiveMHFQ')
        self.assertGreater(summary['total_evaluated_tasks'], 0)
        self.assertGreater(summary['offloading_ratio'], 0.0)
        self.assertTrue(np.all(mes_q >= 0.0))


if __name__ == "__main__":
    unittest.main()
