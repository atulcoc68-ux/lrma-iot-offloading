import os
import glob
import json
import pandas as pd
import numpy as np

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig


class LRMATask:
    """
    Task object holding requirements and attributes extracted from the Alibaba trace.
    Paper Eq: T_{i,k}^t = {size_{i,k}^t, C_{i,k}^t, G_{i,k}^t, R_{i,k}^t}
    size_{i,k}^t = \rho * C_{i,k}^t (Eq. Section III-A)
    """
    def __init__(self, task_id, arrival_slot, cpu_milli, gpu_milli, gpu_spec, duration, rho=EnvConfig.RHO):
        self.task_id = str(task_id)
        self.name = self.task_id
        self.arrival_slot = int(arrival_slot)
        self.start_slot = self.arrival_slot
        self.t_arrival = self.arrival_slot
        
        # CPU requirement C_{i,k}^t (in mega-cycles)
        self.C = float(cpu_milli) if pd.notna(cpu_milli) and float(cpu_milli) > 0 else 1000.0
        # GPU requirement G_{i,k}^t
        self.G = float(gpu_milli) if pd.notna(gpu_milli) and float(gpu_milli) >= 0 else 0.0
        
        # GPU Type R_{i,k}^t \in {0, 1, 2, 3} (Eq. 19f)
        self.gpu_spec_str = str(gpu_spec).strip() if pd.notna(gpu_spec) else ''
        self.R = self._map_gpu_type(self.gpu_spec_str)
        
        # Task duration in seconds
        self.duration = float(duration) if pd.notna(duration) and float(duration) > 0 else 1.0
        
        # Task size size_{i,k}^t = rho * C (bits), constrained by size < omega = 10^8 bits (100 MB)
        computed_size = rho * self.C * 1e3  # Convert milli-CPU cycles to bits
        self.size = min(float(computed_size), EnvConfig.OMEGA - 1.0)
        self.size = max(1e4, self.size)  # Lower bound 10 KB

    def _map_gpu_type(self, spec_str):
        if self.G == 0:
            return 0  # R=0: General CPU / no GPU required
        
        spec_upper = str(spec_str).upper()
        if 'GPUSPEC05' in spec_upper or 'GPUSHARE20' in spec_upper:
            return 1
        elif 'GPUSPEC10' in spec_upper or 'GPUSPEC20' in spec_upper or 'GPUSHARE40' in spec_upper or 'GPUSHARE60' in spec_upper:
            return 2
        elif 'GPUSPEC25' in spec_upper or 'GPUSPEC33' in spec_upper or 'GPUSHARE80' in spec_upper or 'GPUSHARE100' in spec_upper:
            return 3
        else:
            return (hash(self.task_id) % 3) + 1

    def to_dict(self):
        return {
            'task_id': self.task_id,
            'arrival_slot': self.arrival_slot,
            'cpu_milli': self.C,
            'gpu_milli': self.G,
            'gpu_spec': self.gpu_spec_str,
            'gpu_type': self.R,
            'duration': self.duration,
            'size_bits': self.size
        }

    def __repr__(self):
        return f"LRMA_Task(ID={self.task_id}, size={self.size/8e6:.2f}MB, C={self.C}, G={self.G}, R={self.R})"


class AlibabaWorkloadLoader:
    """
    Processes the Alibaba PAI trace for LRMA simulation.
    Supports strict Train/Test workload separation and reproducible slot task sequence generation.
    """
    def __init__(self, pods_file=None, nodes_file=None):
        self.pods_file = pods_file or EnvConfig.PODS_FILE
        self.nodes_file = nodes_file or EnvConfig.NODES_FILE
        self.all_nodes_file = os.path.join(os.path.dirname(self.nodes_file), "openb_node_list_all_node.csv")
        
        self.raw_task_count = 0
        self.cleaned_task_count = 0
        self.total_node_count = 0
        self.gpu_node_count = 0
        
        self.tasks_df = self._load_and_clean_dataset()
        self.train_df, self.test_df = self._create_train_test_split(test_ratio=0.3, seed=42)

    def _load_and_clean_dataset(self):
        # 1. Load Nodes
        if os.path.exists(self.all_nodes_file):
            nodes_all_df = pd.read_csv(self.all_nodes_file)
            self.total_node_count = len(nodes_all_df)
        else:
            self.total_node_count = 1523

        if os.path.exists(self.nodes_file):
            nodes_gpu_df = pd.read_csv(self.nodes_file)
            self.gpu_node_count = len(nodes_gpu_df)
        else:
            self.gpu_node_count = 1213

        # 2. Load Pods/Tasks
        data_dir = os.path.dirname(self.pods_file)
        pod_pattern = os.path.join(data_dir, "openb_pod_list_*.csv")
        all_pod_files = glob.glob(pod_pattern)
        if not all_pod_files:
            all_pod_files = [self.pods_file]

        dfs = []
        for pf in all_pod_files:
            try:
                temp_df = pd.read_csv(pf)
                temp_df['source_file'] = os.path.basename(pf)
                dfs.append(temp_df)
            except Exception:
                pass
        
        combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.read_csv(self.pods_file)
        self.raw_task_count = len(combined_df)

        cleaned_df = combined_df.copy()
        cleaned_df['creation_time'] = pd.to_numeric(cleaned_df['creation_time'], errors='coerce')
        cleaned_df['deletion_time'] = pd.to_numeric(cleaned_df['deletion_time'], errors='coerce')
        cleaned_df = cleaned_df.dropna(subset=['creation_time', 'deletion_time'])
        
        cleaned_df['duration'] = cleaned_df['deletion_time'] - cleaned_df['creation_time']
        cleaned_df = cleaned_df[cleaned_df['duration'] > 0].copy()
        
        cleaned_df = cleaned_df.drop_duplicates(subset=['name', 'source_file']).copy()
        cleaned_df['task_unique_id'] = cleaned_df['name'] + '_' + cleaned_df['source_file']
        cleaned_df['gpu_spec'] = cleaned_df['gpu_spec'].fillna(cleaned_df['source_file'])
        
        min_time = cleaned_df['creation_time'].min()
        cleaned_df['start_slot'] = ((cleaned_df['creation_time'] - min_time) / 10.0).astype(int) % EnvConfig.TOTAL_SLOTS

        self.cleaned_task_count = len(cleaned_df)
        return cleaned_df.reset_index(drop=True)

    def _create_train_test_split(self, test_ratio=0.3, seed=42):
        """Splits cleaned trace into disjoint Train (70%) and Test (30%) sets."""
        shuffled = self.tasks_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = int(len(shuffled) * (1.0 - test_ratio))
        train_df = shuffled.iloc[:split_idx].reset_index(drop=True)
        test_df = shuffled.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df

    def generate_reproducible_slot_workload(self, dataset_split='test', seed=42, num_ed=EnvConfig.NUM_ED,
                                             total_slots=EnvConfig.TOTAL_SLOTS, arrival_rate=0.6):
        """
        Generates and saves deterministic 300-slot task workload sequence for seed.
        Guarantees ALL comparative algorithms receive the IDENTICAL task workload.
        """
        df = self.train_df if dataset_split == 'train' else self.test_df
        rng = np.random.RandomState(seed)
        
        workload_by_slot = {}
        task_pointer = 0
        num_tasks = len(df)

        for t in range(1, total_slots + 1):
            num_generated = min(EnvConfig.MAX_N, max(1, int(rng.binomial(num_ed, arrival_rate))))
            slot_task_list = []
            for _ in range(num_generated):
                row = df.iloc[task_pointer % num_tasks]
                task_pointer += 1
                t_obj = LRMATask(row['task_unique_id'], t, row['cpu_milli'], row['gpu_milli'], row['gpu_spec'], row['duration'])
                slot_task_list.append(t_obj)
            workload_by_slot[t] = slot_task_list

        # Save workload trace to disk for auditability
        save_path = os.path.join(EnvConfig.RAW_RESULTS_DIR, f"workload_trace_{dataset_split}_seed{seed}_N{num_ed}_rate{int(arrival_rate*100)}.json")
        json_data = {
            'seed': seed,
            'dataset_split': dataset_split,
            'num_ed': num_ed,
            'arrival_rate': arrival_rate,
            'total_slots': total_slots,
            'workload': {t: [task.to_dict() for task in tasks] for t, tasks in workload_by_slot.items()}
        }
        with open(save_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        return workload_by_slot

    def generate_verification_report(self):
        """Generates statistical dataset report with Train/Test split verification."""
        report = f"""# Alibaba Cloud Workload Trace Dataset Verification & Train/Test Split Report

## Dataset Summary Metrics
- **Raw Production Tasks Count**: {self.raw_task_count}
- **Cleaned Usable Tasks Count**: {self.cleaned_task_count}
- **Training Tasks Count (70% Split)**: {len(self.train_df)}
- **Evaluation Tasks Count (30% Split - Disjoint)**: {len(self.test_df)}
- **Total Heterogeneous Computing Nodes**: {self.total_node_count}
- **GPU Computing Nodes**: {self.gpu_node_count}

## Task Attribute Distributions (Evaluation Split)
- **GPU Resource-Type Distribution (R in {{0, 1, 2, 3}})**:
  - R=0 (General / CPU): {sum(1 for _, r in self.test_df.iterrows() if r['gpu_milli']==0)} tasks
  - R=1, 2, 3 (Specific GPU Types): {sum(1 for _, r in self.test_df.iterrows() if r['gpu_milli']>0)} tasks
"""
        report_path = os.path.join(EnvConfig.REPORTS_DIR, "dataset_verification.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Generated Dataset Verification Report: {report_path}")
        return report


if __name__ == "__main__":
    loader = AlibabaWorkloadLoader()
    loader.generate_verification_report()
