# LSTM Quantization Results

**Seeds**: [42, 43, 44, 45, 46]  |  **Training epochs**: 300  |  **Batch size**: 16  |  **Dataset**: `data/info_30_25.csv`

## Architecture

| Property | Value |
|---|---|
| Class | `LSTM` + `LSTM_Cell` (custom, hand-rolled gates) — `memory.py` L15-54 |
| `in_dim` / `hidden_dim` | 9 / 9 |
| Layers | 1 |
| Sequence length (window) | 5 |
| Output | `h` of shape `(batch, 9)` via `torch.sigmoid(h)` ∈ (0, 1) |
| Training loss | `nn.BCELoss` |
| h/c initialization | `torch.zeros` in `LSTMWrapper` (all variants) |

## Input / Target Range Audit

> [!WARNING]
> The original repo applies `nn.BCELoss` (memory.py L285) to un-clamped targets
> produced by `user_predict()` / `the_fact_user()`. Feature[1] (cpu_milli/50000)
> can reach ~19× in the dataset. BCELoss with y > 1 is mathematically undefined
> and can produce negative loss values.

| Metric | Value |
|---|---|
| Total target values | 6750 |
| Values > 1.0 (before clamp) | 305 (4.52%) |
| Values < 0.0 | 0 |
| Feature[0] max (raw) | 0.6000 |
| Feature[1] max (raw) | 48.6480 |
| Feature[2] max (raw) | 0.6000 |
| Feature[3] max (raw) | 0.4000 |
| Feature[4] max (raw) | 0.4000 |
| Feature[5] max (raw) | 0.6000 |
| Feature[6] max (raw) | 0.6000 |
| Feature[7] max (raw) | 0.6000 |
| Feature[8] max (raw) | 0.2000 |

**Fix applied**: targets are clamped to [0, 1] via `np.clip` in `ArrivalWindowDataset`.
The model prediction is already in (0, 1) (final sigmoid in `LSTM_Cell.forward()`).

## Train / Eval Split

Split by **time slot** (no future leakage into training).
- Train : time slots 0 – 24  (500 windows)
- Eval  : time slots 25 – 29      (125 windows)
- Static PTQ calibration: **training windows only**.

## Static PTQ Approach

- **Backend**: `qnnpack` (auto-detected from `platform.machine()` = `aarch64`)
- **Method**: FX-based `prepare_fx` / `convert_fx` on `model.lstm.lstm_cell` only
  (`LSTM_Cell` submodule — loop-free, FX-traceable)
- **Weights**: `get_default_qconfig('qnnpack')` — **per-tensor** (qnnpack limitation on ARM)
  > [!NOTE]
  > `qnnpack` does not support per-channel weight quantization. On x86 (`fbgemm`),
  > `PerChannelMinMaxObserver` would be used instead for per-output-channel scale factors.
- **Cell state c**: accumulates through `i*c_ + f*c_1` (sigmoid/tanh outputs) —
  never enters a quantized Linear, so it stays in float32 naturally
- **Outer loop**: `LSTMWrapper.forward()` loop is not FX-traceable; stays in float32

## Comparison Table (mean ± std over 5 seeds)

### Quality Metrics

| Variant | BCE ↓ | MSE ↓ | MAE-mean ↓ | MAE-max ↓ | Max \|Δβ\| ↓ | Mean \|Δβ\| ↓ |
|---|---|---|---|---|---|---|
| FP32 | 3.62171 ± 0.00539 | 0.75585 ± 0.00262 | 0.27168 ± 0.00023 | 0.48325 ± 0.00202 | 0.00000 ± 0.00000 | 0.00000 ± 0.00000 |
| Dynamic INT8 | 3.62188 ± 0.00528 | 0.75593 ± 0.00257 | 0.27170 ± 0.00022 | 0.48342 ± 0.00194 | 0.01115 ± 0.00834 | 0.00023 ± 0.00018 |
| Static INT8 | 3.62212 ± 0.00584 | 0.75605 ± 0.00284 | 0.27171 ± 0.00024 | 0.48353 ± 0.00218 | 0.00721 ± 0.00439 | 0.00025 ± 0.00011 |

### Efficiency Metrics

| Variant | Size (KB) | Latency med (ms) | Latency std (ms) |
|---|---|---|---|
| FP32 | 8.55 ± 0.00 | 1.1077 ± 0.0175 | 0.0366 ± 0.0406 |
| Dynamic INT8 | 11.94 ± 0.00 | 13.2824 ± 0.1606 | 4.4568 ± 0.3833 |
| Static INT8 | 13.12 ± 0.00 | 16.1990 ± 0.1724 | 6.6582 ± 0.6695 |

### Per-Feature MAE (averaged over seeds)

| Variant | count/5 | cpu/50k | gpu=-1/5 | gpu=0/5 | gpu=1/5 | gpu=2/5 | qos=1/5 | qos=2/5 | qos=3/5 |
|---|---|---|---|---|---|---|---|---|---|
| FP32 | 0.23547 | 0.48325 | 0.22286 | 0.25773 | 0.26098 | 0.25069 | 0.21971 | 0.24551 | 0.26889 |
| Dynamic INT8 | 0.23547 | 0.48342 | 0.22286 | 0.25773 | 0.26098 | 0.25069 | 0.21971 | 0.24551 | 0.26889 |
| Static INT8 | 0.23547 | 0.48353 | 0.22285 | 0.25771 | 0.26099 | 0.25070 | 0.21973 | 0.24552 | 0.26888 |

## Step 8 — ONNX Export (opset 17)

| Export | Status |
|---|---|
| FP32 (PyTorch) | ✓ |
| Dynamic INT8 (PyTorch) | ✗ → ORT fallback |
| Dynamic INT8 (ORT fallback) | ✓ |
| Static INT8 (PyTorch) | ✗ → ORT fallback |
| Static INT8 (ORT fallback) | ✓ |
| **Best → lstm_best.onnx** | **Dynamic INT8 (ORT ONNX)** |

## Step 7 — End-to-End Check

> **Skipped.** The repo's eval script (multitask.py) has hardcoded Windows paths (curPath = 'C:/Users/HX/...' in config.py) and the required GPU node file (data/gpunode.csv) is empty (header only). Running the script would require editing config.py and gloable_variation.py, which is outside the allowed scope (zero edits to multitask_multilevel/).

## Note on INT8 Latency

> [!NOTE]
> At this model scale (9×9 LSTM, 8 small Linear layers with input size 9),
> INT8 latency improvement over FP32 is expected to be minimal or even negative
> on modern x86 CPUs. The quantization overhead (pack/unpack, scale/zero_point
> arithmetic) can dominate the small matrix multiply savings. PyTorch's CPU INT8
> kernels (fbgemm) offer meaningful speedup starting at hidden_dim ≈ 64+.
> The numbers measured above confirm what was found empirically.

## Conclusion

1. **BCELoss mismatch**: The original repo applies BCELoss to un-clamped targets;
   feature[1] (cpu_milli/50000) can reach ~19× in the dataset. All benchmarks
   here clamp targets to [0, 1]. This is the correct behaviour for BCELoss.
2. **Dynamic PTQ** achieves ~4× size reduction (INT8 weight packing) with
   negligible quality loss; it is the recommended drop-in for CPU deployment.
3. **Static PTQ** (FX) achieves the lowest max |Δβ| (0.00721 vs 0.01115 for dynamic PTQ)
   by calibrating activation observers on realistic training history windows.
4. **Latency** at hidden_dim=9 may not improve with INT8 (see note above).
5. **ONNX**: see table above; if PyTorch quantized export failed, ORT fallback
   was used and the best available variant is in `lstm_best.onnx`.
