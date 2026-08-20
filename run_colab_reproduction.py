import os
import sys
import time
import torch
import numpy as np
import pandas as pd

# 1. Environment & CUDA Detection
print("=" * 70)
print("GOOGLE COLAB / LOCAL LRMA REPRODUCTION RUNNER")
print("=" * 70)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Detected Compute Device: {device.upper()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")

# 2. Add source directory to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.config import EnvConfig
from src.data_loader import AlibabaWorkloadLoader
from train import train_lrma_agent
from run_experiments import run_all_experiments
from generate_paper_plots import main as generate_all_plots


def main():
    print("\n--- Step 1: Validating Alibaba Workload Dataset ---")
    loader = AlibabaWorkloadLoader()
    print(loader.generate_verification_report())

    print("\n--- Step 2: Running LRMA Simulation & Benchmark Sweeps ---")
    run_all_experiments(seeds=[42], slots=100)

    print("\n--- Step 3: Generating Real Figures 4-10 from Raw Results ---")
    generate_all_plots()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS & FIGURE GENERATIONS COMPLETED SUCCESSFULLY!")
    print(f"Results saved in: {EnvConfig.FIGURES_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
