import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd

from src.config import EnvConfig
from src.data_loader import AlibabaWorkloadLoader
from src.lstm_model import WorkloadPredictor, quantize_predictor
from evaluate import evaluate_policy

def benchmark_lstm_models():
    print("=" * 70)
    print("STEP 1: BENCHMARKING LSTM MODEL: FP32 vs DYNAMIC INT8 QUANTIZATION")
    print("=" * 70)

    # 1. Instantiate models
    torch.manual_seed(42)
    np.random.seed(42)
    fp32_model = WorkloadPredictor(input_dim=4, hidden_dim=64, output_dim=4, num_layers=2)
    fp32_model.eval()

    int8_model = quantize_predictor(fp32_model)
    int8_model.eval()

    # 2. Model Size on Disk
    fp32_path = "/tmp/lstm_fp32.pt"
    int8_path = "/tmp/lstm_int8.pt"
    torch.save(fp32_model.state_dict(), fp32_path)
    torch.save(int8_model.state_dict(), int8_path)
    fp32_size_kb = os.path.getsize(fp32_path) / 1024.0
    int8_size_kb = os.path.getsize(int8_path) / 1024.0
    size_reduction_pct = (1.0 - int8_size_kb / fp32_size_kb) * 100.0

    print(f"FP32 Model Size:      {fp32_size_kb:.2f} KB")
    print(f"Quantized INT8 Size:  {int8_size_kb:.2f} KB  ({size_reduction_pct:.1f}% reduction)")

    # 3. Numerical Precision & Output Error
    num_samples = 1000
    test_inputs = torch.randn(num_samples, EnvConfig.SEQ_LENGTH, 4)

    with torch.no_grad():
        preds_fp32 = fp32_model(test_inputs).numpy()
        preds_int8 = int8_model(test_inputs).numpy()

    abs_diff = np.abs(preds_fp32 - preds_int8)
    mse = float(np.mean((preds_fp32 - preds_int8) ** 2))
    mae = float(np.mean(abs_diff))
    max_diff = float(np.max(abs_diff))
    corrs = [np.corrcoef(preds_fp32[:, col], preds_int8[:, col])[0, 1] for col in range(4)]
    mean_corr = float(np.mean(corrs))

    print(f"\nNumerical Fidelity:")
    print(f"  MSE:                {mse:.6e}")
    print(f"  MAE:                {mae:.6e}")
    print(f"  Max Absolute Error: {max_diff:.6e}")
    print(f"  Output Correlation: {mean_corr:.6f}")

    # 4. Latency Benchmark (1,000 iterations per model)
    # Warmup
    for _ in range(50):
        _ = fp32_model(test_inputs[:1])
        _ = int8_model(test_inputs[:1])

    latencies_fp32 = []
    latencies_int8 = []

    for i in range(1000):
        sample = test_inputs[i : i + 1]

        t0 = time.perf_counter()
        _ = fp32_model(sample)
        latencies_fp32.append((time.perf_counter() - t0) * 1000.0)  # ms

        t0 = time.perf_counter()
        _ = int8_model(sample)
        latencies_int8.append((time.perf_counter() - t0) * 1000.0)  # ms

    med_fp32 = float(np.median(latencies_fp32))
    mean_fp32 = float(np.mean(latencies_fp32))
    std_fp32 = float(np.std(latencies_fp32))

    med_int8 = float(np.median(latencies_int8))
    mean_int8 = float(np.mean(latencies_int8))
    std_int8 = float(np.std(latencies_int8))

    print(f"\nInference Latency (over 1,000 runs):")
    print(f"  FP32 Median Latency:     {med_fp32:.4f} ms (Mean: {mean_fp32:.4f} ± {std_fp32:.4f} ms)")
    print(f"  Quantized INT8 Median:   {med_int8:.4f} ms (Mean: {mean_int8:.4f} ± {std_int8:.4f} ms)")

    # Clean up temp files
    if os.path.exists(fp32_path): os.remove(fp32_path)
    if os.path.exists(int8_path): os.remove(int8_path)

    # 5. End-to-End Simulation Comparison
    print("\n" + "=" * 70)
    print("STEP 2: END-TO-END LRMA SIMULATION: FP32 vs QUANTIZED INT8 LSTM")
    print("=" * 70)

    total_slots = 50  # 50 slots for rapid, accurate end-to-end comparison
    t0 = time.time()
    res_fp32, _, _ = evaluate_policy(
        algorithm='LRMA', total_slots=total_slots, calibrated=True, use_quantized_lstm=False, seed=42
    )
    sim_time_fp32 = time.time() - t0

    t0 = time.time()
    res_int8, _, _ = evaluate_policy(
        algorithm='LRMA', total_slots=total_slots, calibrated=True, use_quantized_lstm=True, seed=42
    )
    sim_time_int8 = time.time() - t0

    print("\nEnd-to-End Simulation Results (50 slots, Calibrated Alibaba Workload):")
    metrics = [
        ('all_task_completion_delay', 'All-Task Completion Delay (s)', '{:.2f}'),
        ('avg_task_completion_delay', 'Average Task Delay (s)', '{:.2f}'),
        ('offloading_ratio', 'Task Offloading Ratio', '{:.4f}'),
        ('avg_energy_consumption', 'Average Energy Consumption (J)', '{:.2f}'),
        ('ed_queue_mean_bits', 'Average ED Queue Size (bits)', '{:.2f}'),
        ('mes_queue_mean_bits', 'Average MES Queue Size (bits)', '{:.2f}'),
        ('total_evaluated_tasks', 'Total Evaluated Tasks', '{:d}')
    ]

    comp_data = []
    for key, name, fmt in metrics:
        v_fp32 = res_fp32[key]
        v_int8 = res_int8[key]
        comp_data.append({
            'Metric': name,
            'FP32 LSTM': fmt.format(v_fp32),
            'Quantized INT8 LSTM': fmt.format(v_int8),
            'Difference': fmt.format(abs(v_fp32 - v_int8))
        })

    df = pd.DataFrame(comp_data)
    print(df.to_string(index=False))
    print(f"\nSimulation Runtime: FP32 = {sim_time_fp32:.2f}s | INT8 = {sim_time_int8:.2f}s")

    # Save results to JSON
    summary_out = {
        'model_level': {
            'fp32_size_kb': fp32_size_kb,
            'int8_size_kb': int8_size_kb,
            'size_reduction_pct': size_reduction_pct,
            'mse': mse,
            'mae': mae,
            'max_diff': max_diff,
            'correlation': mean_corr,
            'fp32_latency_ms': med_fp32,
            'int8_latency_ms': med_int8
        },
        'sim_level': {
            'fp32': res_fp32,
            'int8': res_int8,
            'sim_time_fp32': sim_time_fp32,
            'sim_time_int8': sim_time_int8
        }
    }
    with open('/home/Jetson-Thor/.gemini/antigravity/brain/2f1c9b15-6293-4981-b092-4fff834c774d/scratch/quant_comparison_results.json', 'w') as f:
        json.dump(summary_out, f, indent=2)

if __name__ == "__main__":
    benchmark_lstm_models()
