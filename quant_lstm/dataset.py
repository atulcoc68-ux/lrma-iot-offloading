"""
dataset.py
==========
Replicates user_predict(time, user) from
  multitask_multilevel/multitask.py  (lines 184-197)

Loads data/info_30_25.csv and builds sliding windows of length 5.
The normalization arithmetic is IDENTICAL to the original code.

Feature vector per time-slot (dim = 9)  — from user_predict():
  [0]   task count / 5                          (row shape[0] / 5)
  [1]   sum(cpu_milli) / 50000
  [2]   count(gpu_spec == -1) / 5
  [3]   count(gpu_spec ==  0) / 5
  [4]   count(gpu_spec ==  1) / 5
  [5]   count(gpu_spec ==  2) / 5
  [6]   count(qos == 1) / 5
  [7]   count(qos == 2) / 5
  [8]   count(qos == 3) / 5

Window x : slots [t-4 .. t]  shape (5, 9)
Target  y : slot  [t+1]       shape (9,)

BCELoss range audit
-------------------
The model's sigmoid output is always in (0, 1).  However the TARGET values
produced by the identical arithmetic in the_fact_user() / user_predict() can
exceed 1.0 for:
  • feature[0]: count / 5 — exceeds 1 when >5 tasks for a (user, time) pair
  • feature[1]: sum(cpu_milli) / 50000 — easily exceeds 1 (cpu_milli up to
    ~950 000 per task; even one task gives 950000/50000 = 19.0)

This is a bug in the original repo's BCELoss training (memory.py L285).
BCELoss with y > 1 is mathematically undefined and can produce negative loss.

Our resolution (applied in ArrivalWindowDataset):
  1. We log the fraction of target values > 1.0 before clamping.
  2. We clamp targets to [0, 1] with torch.clamp so BCELoss is well-defined.
  3. The model prediction (sigmoid output) is already in (0, 1) — no change.

Time-based train/eval split
---------------------------
We split by time slot, NOT randomly, to reflect realistic deployment:
  • train : time slots [0 .. cutoff_time - 1]
  • eval  : time slots [cutoff_time .. T-1]
Default cutoff_time = 24 (80 % of 30 slots).
Calibration for static PTQ uses TRAINING windows only.
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


# ─────────────────────────────────────────────────────────────────────────────
# Feature construction — identical to user_predict() in multitask.py
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_vector(df_slot: pd.DataFrame) -> np.ndarray:
    """
    Reproduce the arithmetic of user_predict() for one (user, time) slice.
    Returns a float32 array of shape (9,).

    NOTE: feature[1] (cpu_milli sum / 50000) can exceed 1.0.
    Clamping is applied at the Dataset level, not here, so we can audit
    the raw distribution before clamping.
    """
    vec = np.zeros(9, dtype=np.float32)
    if df_slot.empty:
        return vec
    vec[0] = len(df_slot) / 5.0
    vec[1] = float(df_slot["cpu_milli"].sum()) / 50000.0
    for gpu_spec_idx, gpu_spec_val in enumerate([-1, 0, 1, 2]):
        vec[2 + gpu_spec_idx] = (df_slot["gpu_spec"] == gpu_spec_val).sum() / 5.0
    for qos_idx, qos_val in enumerate([1, 2, 3]):
        vec[6 + qos_idx] = (df_slot["qos"] == qos_val).sum() / 5.0
    return vec


def build_feature_table(csv_path: str):
    """
    Build feature matrix F of shape (n_users, n_times, 9).
    Returns F, sorted user list, sorted time list.
    """
    df = pd.read_csv(csv_path)
    users = sorted(df["from"].unique().tolist())
    times = sorted(df["time"].unique().tolist())
    F = np.zeros((len(users), len(times), 9), dtype=np.float32)
    for u_idx, user in enumerate(users):
        df_user = df[df["from"] == user]
        for t_idx, t in enumerate(times):
            df_slot = df_user[df_user["time"] == t]
            F[u_idx, t_idx] = _build_feature_vector(df_slot)
    return F, users, times


def audit_target_range(F: np.ndarray) -> dict:
    """
    Examine the feature matrix for values outside [0, 1].
    Returns a summary dict with fractions and per-feature max.
    """
    total = F.size
    above_one = (F > 1.0).sum()
    below_zero = (F < 0.0).sum()
    per_feature_max = F.max(axis=(0, 1))  # shape (9,)
    return {
        "total_values": int(total),
        "above_one_count": int(above_one),
        "above_one_frac": float(above_one / total),
        "below_zero_count": int(below_zero),
        "per_feature_max": per_feature_max.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ArrivalWindowDataset(Dataset):
    """
    Sliding-window dataset over all users and all eligible time-slots.

    x : torch.FloatTensor of shape (5, 9)   — history window [t-4 .. t]
    y : torch.FloatTensor of shape (9,)     — next slot [t+1], CLAMPED to [0,1]

    Each sample also stores the time index of the TARGET slot, enabling
    time-based train/eval splitting without breaking into sub-Datasets.
    """

    WINDOW = 5  # matches user_predict()'s range(time-4, time+1)

    def __init__(self, csv_path: str, verbose: bool = True):
        self.F, self.users, self.times = build_feature_table(csv_path)
        n_users, n_times, _ = self.F.shape

        # Audit before clamping
        audit = audit_target_range(self.F)
        self._audit = audit
        if verbose:
            print(f"[Dataset] Raw feature range audit:")
            print(f"          total values   : {audit['total_values']}")
            print(f"          above 1.0      : {audit['above_one_count']} "
                  f"({100*audit['above_one_frac']:.2f}%) — BCELoss mismatch")
            print(f"          below 0.0      : {audit['below_zero_count']}")
            print(f"          per-feature max: "
                  + ", ".join(f"f{i}={v:.3f}" for i, v in enumerate(audit['per_feature_max'])))
            if audit['above_one_frac'] > 0:
                print(f"          [WARN] Targets clamped to [0,1] for BCELoss validity.")
                print(f"          [WARN] The original repo uses these un-clamped values in")
                print(f"          [WARN] nn.BCELoss (memory.py L285), which is incorrect.")

        # Build (x, y, target_time) triples
        self.samples = []       # (x: np.ndarray, y: np.ndarray, target_time_idx: int)
        for u in range(n_users):
            for t in range(self.WINDOW - 1, n_times - 1):
                x = self.F[u, t - self.WINDOW + 1 : t + 1].copy()   # (5, 9)
                y_raw = self.F[u, t + 1].copy()                       # (9,)
                y = np.clip(y_raw, 0.0, 1.0)                          # clamp
                target_time_idx = t + 1   # index into self.times list
                self.samples.append((x, y, target_time_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, _ = self.samples[idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

    def get_target_time_indices(self) -> list:
        """Return list of target_time_idx for each sample (for time-based split)."""
        return [s[2] for s in self.samples]


# ─────────────────────────────────────────────────────────────────────────────
# Time-based split
# ─────────────────────────────────────────────────────────────────────────────

def split_by_time(dataset: ArrivalWindowDataset,
                  cutoff_time_idx: int = None,
                  train_frac: float = 0.8
                  ) -> tuple:
    """
    Split dataset into train / eval subsets by time slot index.

    The LAST (1-train_frac) fraction of time indices goes to eval.
    No sample from a future time slot leaks into training.

    Args
    ----
    dataset          : ArrivalWindowDataset
    cutoff_time_idx  : first time index that goes to eval.
                       If None, computed from train_frac.
    train_frac       : fraction of time slots to use for training.

    Returns
    -------
    (train_subset, eval_subset, cutoff_time_idx)
    """
    time_indices = dataset.get_target_time_indices()
    unique_times = sorted(set(time_indices))
    n_times = len(unique_times)

    if cutoff_time_idx is None:
        cutoff_pos = int(np.ceil(train_frac * n_times))
        cutoff_time_idx = unique_times[cutoff_pos] if cutoff_pos < n_times else unique_times[-1]

    train_ids = [i for i, t in enumerate(time_indices) if t < cutoff_time_idx]
    eval_ids  = [i for i, t in enumerate(time_indices) if t >= cutoff_time_idx]

    train_subset = Subset(dataset, train_ids)
    eval_subset  = Subset(dataset, eval_ids)
    return train_subset, eval_subset, cutoff_time_idx


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def get_default_csv_path(data_dir: str = None) -> str:
    """Locate info_30_25.csv relative to quant_lstm/ or a caller-supplied dir."""
    if data_dir is not None:
        p = os.path.join(data_dir, "info_30_25.csv") if not data_dir.endswith(".csv") else data_dir
        if os.path.isfile(p):
            return p
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "lrma_repo", "multitask_multilevel", "data", "info_30_25.csv"),
        os.path.join(here, "..", "multitask_multilevel", "data", "info_30_25.csv"),
        os.path.join(here, "data", "info_30_25.csv"),
        os.path.join("lrma_repo", "multitask_multilevel", "data", "info_30_25.csv"),
        os.path.join("multitask_multilevel", "data", "info_30_25.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return os.path.join(here, "..", "lrma_repo", "multitask_multilevel", "data", "info_30_25.csv")

