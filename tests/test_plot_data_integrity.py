import os
import glob
import unittest
import numpy as np
import pandas as pd
from src.config import EnvConfig
from generate_paper_plots import validate_queue_files, EXPECTED_SEEDS

class TestPlotDataIntegrity(unittest.TestCase):

    def test_fig5_has_35_files(self):
        """Verify Fig 5 raw queue files exist for 7 V values * 5 seeds = 35 files."""
        v_vals = [1, 10, 20, 30, 40, 50, 100]
        files = []
        for v in v_vals:
            for seed in EXPECTED_SEEDS:
                fpath = os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig5_queues_V{v}_seed{seed}.csv")
                files.append(fpath)
        self.assertEqual(len(files), 35)
        dfs = validate_queue_files(files)
        self.assertEqual(len(dfs), 35)

    def test_fig7_has_10_files(self):
        """Verify Fig 7 raw queue files exist for 2 reset modes * 5 seeds = 10 files."""
        files = []
        for r_mode in ["resetTrue", "resetFalse"]:
            for seed in EXPECTED_SEEDS:
                fpath = os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig7_queues_{r_mode}_seed{seed}.csv")
                files.append(fpath)
        self.assertEqual(len(files), 10)
        dfs = validate_queue_files(files)
        self.assertEqual(len(dfs), 10)

    def test_fig8_has_45_files(self):
        """Verify Fig 8 raw queue files exist for 3 algorithms * 3 ED counts * 5 seeds = 45 files."""
        q_algs = ["FCFS", "MMC", "MHFQ"]
        n_vals = [20, 25, 30]
        files = []
        for q in q_algs:
            for n in n_vals:
                for seed in EXPECTED_SEEDS:
                    fpath = os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig8_queues_{q}_N{n}_seed{seed}.csv")
                    files.append(fpath)
        self.assertEqual(len(files), 45)
        dfs = validate_queue_files(files)
        self.assertEqual(len(dfs), 45)

    def test_fig10_has_60_files(self):
        """Verify Fig 10 raw queue files exist for 4 algorithms * 3 arrival rates * 5 seeds = 60 files."""
        algs = ["LRMA", "MA3MCO", "L-MADDPG", "DVCCO"]
        rates = ["40pct", "60pct", "80pct"]
        files = []
        for alg in algs:
            for rate in rates:
                for seed in EXPECTED_SEEDS:
                    fpath = os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig10_queues_{alg}_{rate}_seed{seed}.csv")
                    files.append(fpath)
        self.assertEqual(len(files), 60)
        dfs = validate_queue_files(files)
        self.assertEqual(len(dfs), 60)

    def test_fig10_filename_uses_pct(self):
        """Verify Fig 10 filenames use 'pct' (e.g. 40pct) and not '%' (40%)."""
        for alg in ["LRMA", "MA3MCO", "L-MADDPG", "DVCCO"]:
            for r in ["40pct", "60pct", "80pct"]:
                for seed in EXPECTED_SEEDS:
                    fpath = os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig10_queues_{alg}_{r}_seed{seed}.csv")
                    self.assertTrue(os.path.exists(fpath), f"File {fpath} with 'pct' suffix must exist")
                    percent_path = fpath.replace("pct", "%")
                    self.assertFalse(os.path.exists(percent_path), f"File with literal '%' {percent_path} should not exist")

    def test_expected_seeds_present(self):
        """Verify exactly seeds 42, 43, 44, 45, 46 are present across all queue files."""
        self.assertEqual(sorted(EXPECTED_SEEDS), [42, 43, 44, 45, 46])

    def test_queue_columns_exist(self):
        """Verify required columns ('slot', 'ed_queue_bits', 'mes_queue_bits') exist in all raw queue CSV files."""
        sample_file = os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig5_queues_V20_seed42.csv")
        df = pd.read_csv(sample_file)
        for col in ['slot', 'ed_queue_bits', 'mes_queue_bits']:
            self.assertIn(col, df.columns)

    def test_slot_axes_match(self):
        """Verify all queue files have matching slot axes."""
        f1 = os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig5_queues_V20_seed42.csv")
        f2 = os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig10_queues_LRMA_60pct_seed44.csv")
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
        np.testing.assert_array_equal(df1['slot'].values, df2['slot'].values)

    def test_no_nan_or_inf(self):
        """Verify no NaN or Inf values exist in raw queue files."""
        files = [
            os.path.join(EnvConfig.RAW_RESULTS_DIR, f"fig10_queues_LRMA_60pct_seed{s}.csv")
            for s in EXPECTED_SEEDS
        ]
        for f in files:
            df = pd.read_csv(f)
            self.assertFalse(df['ed_queue_bits'].isnull().any())
            self.assertFalse(df['mes_queue_bits'].isnull().any())
            self.assertFalse(np.isinf(df['ed_queue_bits']).any())
            self.assertFalse(np.isinf(df['mes_queue_bits']).any())

    def test_no_missing_queue_files(self):
        """Verify total raw queue files count equals 35 + 10 + 45 + 60 = 150 files."""
        total_expected = 35 + 10 + 45 + 60
        fig5_files = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig5_queues_*.csv"))
        fig7_files = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig7_queues_*.csv"))
        fig8_files = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig8_queues_*.csv"))
        fig10_files = glob.glob(os.path.join(EnvConfig.RAW_RESULTS_DIR, "fig10_queues_*.csv"))

        self.assertEqual(len(fig5_files), 35)
        self.assertEqual(len(fig7_files), 10)
        self.assertEqual(len(fig8_files), 45)
        self.assertEqual(len(fig10_files), 60)
        self.assertEqual(len(fig5_files) + len(fig7_files) + len(fig8_files) + len(fig10_files), total_expected)

if __name__ == "__main__":
    unittest.main()
