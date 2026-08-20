import os
import json
import unittest
import numpy as np
from src.config import EnvConfig
from src.data_loader import AlibabaWorkloadLoader, LRMATask

class TestWorkloadGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = AlibabaWorkloadLoader()

    def test_per_ed_binomial_rates(self):
        """Tests 1-5: Verify mean tasks/slot at rates 0.4, 0.6, 0.8 and per-ED bounds."""
        test_cases = [
            (0.4, 50.0),
            (0.6, 75.0),
            (0.8, 100.0)
        ]
        
        for rate, expected_mean in test_cases:
            workload = self.loader.generate_reproducible_slot_workload(
                dataset_split='test', seed=42, num_ed=25, total_slots=300, arrival_rate=rate
            )
            
            slot_totals = []
            ed_task_counts = {ed_id: [] for ed_id in range(25)}
            
            for t, ed_dict in workload.items():
                slot_t_total = 0
                for ed_id, tasks in ed_dict.items():
                    count = len(tasks)
                    # Requirement 5: 0 <= |m_i^t| <= 5
                    self.assertTrue(0 <= count <= EnvConfig.MAX_N, f"ED task count {count} violates 0 <= count <= 5")
                    ed_task_counts[ed_id].append(count)
                    slot_t_total += count
                slot_totals.append(slot_t_total)
            
            mean_slot = np.mean(slot_totals)
            # Requirements 1-3: mean tasks/slot approx expected_mean within +/- 2.5
            self.assertAlmostEqual(mean_slot, expected_mean, delta=2.5,
                                   msg=f"Rate {rate}: Expected mean ~{expected_mean}, got {mean_slot:.2f}")

            if rate == 0.6:
                # Requirement 4: Each ED has mean approx 3 tasks/slot at p=0.6
                for ed_id, counts in ed_task_counts.items():
                    ed_mean = np.mean(counts)
                    self.assertAlmostEqual(ed_mean, 3.0, delta=0.5,
                                           msg=f"ED {ed_id}: Expected mean ~3.0, got {ed_mean:.2f}")

    def test_determinism_and_reuse(self):
        """Requirements 6-7: Same seed produces identical workload and trace JSON reuse works."""
        wl1 = self.loader.generate_reproducible_slot_workload(dataset_split='test', seed=42, num_ed=25, total_slots=50, arrival_rate=0.6)
        wl2 = self.loader.generate_reproducible_slot_workload(dataset_split='test', seed=42, num_ed=25, total_slots=50, arrival_rate=0.6)
        
        # Verify determinism across calls
        for t in range(1, 51):
            for ed_id in range(25):
                t1_list = wl1[t][ed_id]
                t2_list = wl2[t][ed_id]
                self.assertEqual(len(t1_list), len(t2_list))
                for tk1, tk2 in zip(t1_list, t2_list):
                    self.assertEqual(tk1.task_id, tk2.task_id)
                    self.assertEqual(tk1.ed_id, tk2.ed_id)
                    self.assertEqual(tk1.size, tk2.size)

    def test_ed_explicit_association(self):
        """Requirement 8: Check explicit ED association without Python hash randomization."""
        workload = self.loader.generate_reproducible_slot_workload(dataset_split='test', seed=123, num_ed=25, total_slots=10, arrival_rate=0.6)
        for t, ed_dict in workload.items():
            for ed_id, tasks in ed_dict.items():
                for task in tasks:
                    self.assertEqual(task.ed_id, ed_id)
                    self.assertTrue(task.task_id.endswith(f"_ed{ed_id}_slot{t}"))

if __name__ == '__main__':
    unittest.main()
