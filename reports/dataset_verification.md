# Alibaba Cloud Workload Trace Dataset Verification Report

## Dataset Summary Metrics
- **Raw Production Tasks Count**: 81520
- **Cleaned Usable Tasks Count**: 81508
- **Total Heterogeneous Computing Nodes**: 1523
- **GPU Computing Nodes**: 1213

## Task Attribute Distributions
- **GPU Resource-Type Distribution (R in {0, 1, 2, 3})**:
  - R=0 (General / CPU): 10880 tasks (13.35%)
  - R=1 (GPU Type 1 - V100): 15933 tasks (19.55%)
  - R=2 (GPU Type 2 - T4): 28329 tasks (34.76%)
  - R=3 (GPU Type 3 - P100): 26366 tasks (32.35%)

## Task Size & Resource Statistics
- **Task Size (MB)**: Mean = 9.1785, Std = 3.7780, Min = 1.2500, Max = 12.5000 (Max constraint < 12.5 MB enforced)
- **CPU Demand (milli-cores)**: Mean = 9797.96, Std = 7973.20, Min = 1000.00, Max = 120200.00
- **GPU Demand (milli-cores)**: Mean = 650.22, Std = 357.99, Min = 0.00, Max = 1000.00
- **Task Duration (s)**: Mean = 23693.77, Std = 382206.38, Min = 1.00, Max = 12537496.00
