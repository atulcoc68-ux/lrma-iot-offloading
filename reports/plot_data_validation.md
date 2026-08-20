# Plot Data Validation Report

**Paper Title:** Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems (IEEE TNSE 2025)  
**Target:** Hardening & Verification of Result and Plot Generation Pipeline (`generate_paper_plots.py`)  
**File Location:** `reports/plot_data_validation.md`

---

## 1. Executive Summary

All 5-seed simulation experiments have been validated and confirmed complete. The plot generation pipeline (`generate_paper_plots.py`) has been hardened with strict, non-silent file validation (`validate_queue_files`), exact seed matching (`[42, 43, 44, 45, 46]`), and filename rate mapping (`40pct`, `60pct`, `80pct`). 

No artificial scaling multipliers, hardcoded paper values, or curve-shifting factors are used anywhere in the codebase. All figures are generated directly and honestly from empirical simulation outputs.

---

## 2. Experimental Data Audit & Summary Counts

| Metric / Category | Expected Count | Actual Count Found | Status |
| :--- | :---: | :---: | :---: |
| **Processed Summary CSV Files** | 4 | 4 | PASS |
| **Fig 5 Raw Queue CSV Files** ($7\text{ V} \times 5\text{ Seeds}$) | 35 | 35 | PASS |
| **Fig 7 Raw Queue CSV Files** ($2\text{ Resets} \times 5\text{ Seeds}$) | 10 | 10 | PASS |
| **Fig 8 Raw Queue CSV Files** ($3\text{ Algs} \times 3\text{ Ns} \times 5\text{ Seeds}$) | 45 | 45 | PASS |
| **Fig 10 Raw Queue CSV Files** ($4\text{ Algs} \times 3\text{ Rates} \times 5\text{ Seeds}$) | 60 | 60 | PASS |
| **Total Raw Queue Files Validated** | **150** | **150** | **100% MATCH** |
| **Missing Files** | 0 | 0 | PASS |

---

## 3. Parameter Configurations Validated

- **Seeds Used:** `[42, 43, 44, 45, 46]` (Exactly 5 seeds, no extra or missing seeds)
- **V Values Used:** `[1, 10, 20, 30, 40, 50, 100]` (Fig 4 & Fig 5)
- **Parameter Reset Modes:** `resetTrue` (LRMA), `resetFalse` (No-reset LRMA) (Fig 6 & Fig 7)
- **Queue Frameworks Evaluated:** `FCFS`, `MMC`, `MHFQ` (Fig 8)
- **IoT Device Counts ($N$):** `20`, `25`, `30` (Fig 8)
- **Comparison Algorithms:** `LRMA`, `MA3MCO`, `L-MADDPG`, `DVCCO` (Fig 9 & Fig 10)
- **Task Arrival Rates:** `40pct` (0.4), `60pct` (0.6), `80pct` (0.8) (Fig 9 & Fig 10)
- **Simulation Time Axis:** 300 slots ($t = 0 \dots 299$)
- **Required CSV Columns:** `slot`, `ed_queue_bits`, `mes_queue_bits`
- **Data Quality Check:** `0` NaN values, `0` Inf values across all 150 raw queue CSV files

---

## 4. Fixes Applied to `generate_paper_plots.py`

1. **Figure 10 Filename Resolution**: Fixed mismatch between filename rates (`40pct`, `60pct`, `80pct`) and plot display labels (`40%`, `60%`, `80%`).
2. **Strict Queue File Validation**: Added `validate_queue_files()` helper near top of script to verify file existence, schema, columns, seed matching, and non-empty slot alignment. Replaced silent `if q_files:` checks with explicit exceptions (`FileNotFoundError`, `ValueError`).
3. **No Arbitrary Globs**: Replaced loose `glob.glob("*seed*.csv")` pattern matching with explicit, seed-indexed filepath lists built from `EXPECTED_SEEDS = [42, 43, 44, 45, 46]`.
4. **Honest Labeling**: Labeled shaded error bands and error bars explicitly as **Mean ± Standard Deviation**.
5. **Authoritative Output Location**: All 7 PNG figures are written to `results/figures/`:
   - `results/figures/fig4_v_impact.png`
   - `results/figures/fig5_v_fluctuations.png`
   - `results/figures/fig6_reset_ablation.png`
   - `results/figures/fig7_reset_fluctuations.png`
   - `results/figures/fig8_mhfq_comparison.png`
   - `results/figures/fig9_algorithm_power.png`
   - `results/figures/fig10_algorithm_fluctuations.png`

---

## 5. Verification & Integrity Checklist

- [x] No hardcoded numerical results in plotting scripts
- [x] No scaling multipliers (0.8, 1.2, etc.) or artificial curve shifts
- [x] Figures are generated dynamically from simulation outputs
- [x] Automated test suite `tests/test_plot_data_integrity.py` passes 10/10 tests
