# Alibaba Cloud Workload Trace Dataset Verification & Train/Test Split Report

## Dataset Summary Metrics
- **Raw Production Tasks Count**: 81520
- **Cleaned Usable Tasks Count**: 81508
- **Training Tasks Count (70% Split)**: 57055
- **Evaluation Tasks Count (30% Split - Disjoint)**: 24453
- **Total Heterogeneous Computing Nodes**: 1523
- **GPU Computing Nodes**: 1213

## Task Attribute Distributions (Evaluation Split)
- **GPU Resource-Type Distribution (R in {0, 1, 2, 3})**:
  - R=0 (General / CPU): 3218 tasks
  - R=1, 2, 3 (Specific GPU Types): 21235 tasks
