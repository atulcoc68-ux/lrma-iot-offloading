import os
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class EnvConfig:
    """
    Authoritative Configuration System matching IEEE TNSE 2025 paper Tables I & II.
    "Multi-Agent DRL-Based Large-Scale Heterogeneous Task Offloading for Dynamic IoT Systems"
    """
    # System Architecture Parameters (Table I & II)
    NUM_ED = 25             # ED count N (20, 25, 30 across experiments)
    NUM_MES = 5             # MES / BS count M = 5
    NUM_GPU_TYPES = 3       # GPU resource types |R| = 3 (R = {0, 1, 2, 3})
    TAU = 1.0               # Time slot duration tau = 1 second
    TOTAL_SLOTS = 300       # Simulation time horizon K = 300 seconds
    MAX_N = 5               # Max_n = 5 tasks per ED per slot (Eq. 19g)
    OMEGA = 1e8             # Max task size omega = 10^8 bits (100 MB) (Table II)
    RHO = 10.0              # CPU cycles per bit ratio rho

    # Wireless Communication Parameters (Section III-B & Table II)
    BANDWIDTH = 20e6        # Channel bandwidth B_i^t = 20 MHz
    TRANS_POWER = 0.5       # Transmission power P_i^{tran} = 0.5 W
    SIGMA_SQUARED = 1e-9    # Noise power sigma^2 = -60 dBm = 10^-9 W (Table II)
    CARRIER_FREQ = 2.4e9    # Carrier frequency f_carrier = 2.4 GHz
    ANTENNA_GAIN = 3.0      # Antenna gain A_antenna = 3.0
    PATH_LOSS_EXP = 2.0     # Path loss attenuation index loss_ple = 2.0
    TRANS_OVERHEAD = 1.05   # Information overhead o_u(t) > 1 (Eq. 9)

    # Computational Resource Capacities (Table I & II)
    LOCAL_CPU_CAPACITY = 2.0e9    # f_{i,c}^{local} = 2.0 GHz (cycles/s)
    LOCAL_GPU_CAPACITY = 4.0e9    # f_{i,r,g}^{local} = 4.0 GHz (cycles/s)
    MES_TOTAL_CPU_CAPACITY = 10e9 # f_{j,c}^{es} = 10.0 GHz (cycles/s)
    MES_GPU_CAPACITY = 8.0e9      # f_{j,r,g}^{es} = 8.0 GHz (cycles/s)

    # MHFQ Time Slices (Section III-A.2)
    TAU_VES = [0.1, 0.3, 0.6]     # Time slices {tau_1^ves, tau_2^ves, tau_3^ves} in seconds

    # Lyapunov & Optimization Parameters (Table II & Eq. 40)
    V = 20.0                # Lyapunov penalty parameter V = 20 (Table II)
    V_SWEEP = [1, 10, 20, 30, 40, 50, 100] # V values for Fig 4 & 5
    ALPHA = 0.5             # Comprehensive reward weighting factor alpha = 0.5 (Eq. 40)

    # Multi-Agent Reinforcement Learning Parameters (Algorithm 1)
    DELTA_RESET = 50        # Parameter reset interval delta^reset = 50 slots (Alg 1, line 26)
    XI_SOFT = 0.01          # Soft update factor xi_soft = 0.01 (Eq. 45)
    SEQ_LENGTH = 10         # LSTM sequence length
    BATCH_SIZE = 64         # Batch size B = 64 (Alg 1, line 24)
    UPDATE_INTERVAL = 64    # Soft update interval delta^soft = 64 slots
    LR_ACTOR = 0.0003       # Learning rate for Actor networks
    LR_CRITIC = 0.001       # Learning rate for Critic network
    NUM_CANDIDATES_P = 5    # Point-to-uniform variation action candidate count P

    BASE_DIR = BASE_DIR
    RAW_RESULTS_DIR = os.path.join(BASE_DIR, "results", "raw")
    PROCESSED_RESULTS_DIR = os.path.join(BASE_DIR, "results", "processed")
    FIGURES_DIR = os.path.join(BASE_DIR, "results", "figures")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")
    CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

    os.makedirs(RAW_RESULTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

    @staticmethod
    def _find_pod_file():
        local_pods = glob.glob(os.path.join(BASE_DIR, "data", "Original dataset", "openb_pod_list_*.csv"))
        if local_pods:
            return local_pods[0]
        colab_pods = glob.glob("/content/openb_pod_list_*.csv")
        if colab_pods:
            return colab_pods[0]
        return os.path.join(BASE_DIR, "data", "Original dataset", "openb_pod_list_gpuspec33.csv")

    @staticmethod
    def _find_node_file():
        local_nodes = glob.glob(os.path.join(BASE_DIR, "data", "Original dataset", "openb_node_list_gpu_node.csv"))
        if local_nodes:
            return local_nodes[0]
        colab_nodes = glob.glob("/content/openb_node_list_gpu_node.csv")
        if colab_nodes:
            return colab_nodes[0]
        return os.path.join(BASE_DIR, "data", "Original dataset", "openb_node_list_gpu_node.csv")

    PODS_FILE = _find_pod_file()
    NODES_FILE = _find_node_file()




