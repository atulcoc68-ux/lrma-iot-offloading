import numpy as np

try:
    from src.config import EnvConfig
    from src.lyapunov import MHFQ, FCFSQueue, MMCQueue, LRMARewardCalculator
    from src.wireless import WirelessModel
    from src.lstm_model import get_future_workload_estimate
except ModuleNotFoundError:
    from config import EnvConfig
    from lyapunov import MHFQ, FCFSQueue, MMCQueue, LRMARewardCalculator
    from wireless import WirelessModel
    from lstm_model import get_future_workload_estimate


class LRMA_Environment:
    """
    IEEE TNSE 2025 Multi-Agent IoT Offloading Environment (Section III, IV, V).
    Models dynamic IoT system over time horizon K = 300 seconds (time slots t = 1 ... 300).
    """
    def __init__(self, loader, predictor, config=EnvConfig, queue_type='MHFQ', num_ed=None, V_val=None):
        self.loader = loader
        self.predictor = predictor
        self.config = config
        self.num_ed = num_ed if num_ed is not None else config.NUM_ED
        self.num_mes = config.NUM_MES
        self.queue_type = queue_type
        
        # Instantiate Queue Framework
        if queue_type == 'FCFS':
            self.mhfq = FCFSQueue(self.num_mes)
        elif queue_type == 'MMC':
            self.mhfq = MMCQueue(self.num_mes)
        else:
            self.mhfq = MHFQ(self.num_mes, config.NUM_GPU_TYPES)

        self.wireless = WirelessModel()
        self.reward_calc = LRMARewardCalculator(V_penalty=V_val if V_val is not None else config.V)

        # Dynamic System Queues (Paper Eq. 1 - 4)
        # Q_device_i(t) for each ED i (Paper Eq. 1-2)
        self.q_device = np.zeros(self.num_ed, dtype=np.float64)
        # Q_es_j(t) for each MES j (Paper Eq. 3-4)
        self.q_es = np.zeros(self.num_mes, dtype=np.float64)
        
        # Static BS and ED positions
        np.random.seed(42)
        self.ed_positions = np.random.uniform(0, 500, size=(self.num_ed, 2))
        self.bs_positions = np.array([
            [100, 100], [400, 100], [250, 250], [100, 400], [400, 400]
        ], dtype=np.float64)

        self.current_time_slot = 0
        self.history_arrival_states = []

    def get_ed_state(self, ed_idx, task, pending_tasks):
        r"""
        Constructs observable local state S_{i,k}(t) for ED agent i (Paper Eq. 34).
        S_{i,k}(t) = { T_{i,k}^t, \widetilde{T}_{i,k-}^t, Q_device_i(t), \sum_{j=1}^n Q_es_j(t), \beta_i^{t+1} }
        """
        # Paper Eq. (34): T_{i,k}^t state features
        t_size = task.size / 8e6  # MB
        t_c = task.C / 1000.0
        t_g = task.G / 1000.0
        t_r = float(task.R)

        # Pending decision tasks state \widetilde{T}_{i,k-}^t
        pending_count = len(pending_tasks)
        pending_size = sum(t_item.size for t_item in pending_tasks) / 8e6

        # Local backlog Q_device_i(t)
        q_dev = float(self.q_device[ed_idx]) / 8e6

        # Neighboring MES backlogs \sum_{j=1}^n Q_es_j(t)
        q_es_sum = float(np.sum(self.q_es)) / 8e6

        # LSTM predicted future arrival state \beta_i^{t+1}
        beta_hat = get_future_workload_estimate(self.predictor, self.history_arrival_states)

        state = np.array([
            t_size, t_c, t_g, t_r,
            pending_count, pending_size,
            q_dev, q_es_sum,
            beta_hat[0] if isinstance(beta_hat, (list, np.ndarray)) else float(beta_hat)
        ], dtype=np.float32)

        return state

    def get_cloud_state(self, task, offloaded_tasks, selected_mes_idx=0):
        r"""
        Constructs observable state S_{i,k}^{cloud}(t) for Cloud agent (Paper Eq. 35).
        S_{i,k}^{cloud}(t) = { T_{i,k}^t, \varsigma_{k-}^t, {Q_es_j(t)}_{j=1}^n, \beta_{cloud}^{t+1} }
        """
        # Paper Eq. (35): Task and MES states
        t_size = task.size / 8e6
        t_c = task.C / 1000.0
        t_g = task.G / 1000.0
        t_r = float(task.R)

        offloaded_count = len(offloaded_tasks)
        mes_queues = [float(self.q_es[j]) / 8e6 for j in range(self.num_mes)]

        beta_hat = get_future_workload_estimate(self.predictor, self.history_arrival_states)

        state = np.array(
            [t_size, t_c, t_g, t_r, offloaded_count] + mes_queues + [beta_hat[0] if isinstance(beta_hat, (list, np.ndarray)) else float(beta_hat)],
            dtype=np.float32
        )
        return state

    def step_task_offloading(self, ed_idx, task, ed_action, cloud_action=None):
        r"""
        Executes single task processing step according to Paper Eq (6)-(18).
        ed_action x_{i,k}^t \in {0, 1}: 0 = local, 1 = offload.
        cloud_action y_{i,k}^t \in {0...M-1}: MES node assignment.
        """
        x_offload = 1 if ed_action > 0 else 0
        
        if x_offload == 0:
            # Paper Eq. (11)-(13): Local execution on ED i
            c_time = (task.C * 1e6) / self.config.LOCAL_CPU_CAPACITY
            g_time = (task.G * 1e6) / self.config.LOCAL_GPU_CAPACITY if task.R > 0 else 0.0
            d_ed = max(c_time, g_time)
            w_ed = float(self.q_device[ed_idx]) / (self.config.LOCAL_CPU_CAPACITY / 8e6 + 1e-5)
            
            total_delay = d_ed + w_ed
            energy = 0.5 * (task.size / 8e6)  # Local energy consumption
            
            # Update local queue (Paper Eq. 1-2)
            self.q_device[ed_idx] = max(0.0, self.q_device[ed_idx] + task.size - self.config.LOCAL_CPU_CAPACITY * self.config.TAU / 10.0)
            
            return {
                'is_offloaded': False,
                'mes_assigned': -1,
                'delay': total_delay,
                'energy': energy,
                'task_size': task.size
            }

        else:
            # Offloading to MES (Paper Eq. 6-10, 14-18)
            mes_idx = cloud_action if cloud_action is not None and 0 <= cloud_action < self.num_mes else (ed_action - 1) % self.num_mes
            
            # Distance and Shannon wireless transmission (Paper Eq. 6-9)
            dist = self.wireless.calculate_distance(self.ed_positions[ed_idx], self.bs_positions[mes_idx])
            h_gain = self.wireless.calculate_channel_gain(dist)
            v_rate = self.wireless.calculate_rate(h_gain)
            t_trans = self.wireless.calculate_transmission_delay(task.size, v_rate)
            to_entry = self.wireless.calculate_offload_completion_entry_time(self.current_time_slot, t_trans)

            # MES Queue execution & waiting time (Paper Eq. 14-18)
            d_bs, w_bs, completion_time = self.mhfq.process_offloaded_task(mes_idx, task, to_entry)
            
            total_delay = t_trans + w_bs + d_bs
            energy = 0.2 * (task.size / 8e6)  # Transmission energy consumption

            # Update MES queue (Paper Eq. 3-4)
            self.q_es[mes_idx] = max(0.0, self.q_es[mes_idx] + task.size - self.config.MES_TOTAL_CPU_CAPACITY * self.config.TAU / 10.0)

            return {
                'is_offloaded': True,
                'mes_assigned': mes_idx,
                'delay': total_delay,
                'energy': energy,
                'task_size': task.size
            }

    def update_time_slot(self, slot_idx, slot_tasks):
        """Advance simulation time slot t (Paper Eq. 1-4)."""
        self.current_time_slot = slot_idx
        
        # Flatten tasks if passed as per-ED dict: {ed_id: [tasks]}
        if isinstance(slot_tasks, dict):
            all_tasks = [t for ed_list in slot_tasks.values() for t in ed_list]
        elif isinstance(slot_tasks, list):
            all_tasks = slot_tasks
        else:
            all_tasks = []

        # Track historical arrival vector \widetilde{T}^t = [task_count, avg_size, avg_C, avg_G]
        if all_tasks:
            avg_sz = np.mean([t.size for t in all_tasks])
            avg_c = np.mean([t.C for t in all_tasks])
            avg_g = np.mean([t.G for t in all_tasks])
            state_vec = np.array([len(all_tasks), avg_sz / 8e6, avg_c / 1000.0, avg_g / 1000.0], dtype=np.float32)
        else:
            state_vec = np.zeros(4, dtype=np.float32)

        self.history_arrival_states.append(state_vec)

        # Decay queues per slot duration \tau = 1s (Paper Eq. 1-4)
        for i in range(self.num_ed):
            self.q_device[i] = max(0.0, self.q_device[i] - self.config.LOCAL_CPU_CAPACITY * self.config.TAU / 10.0)
        for j in range(self.num_mes):
            self.q_es[j] = max(0.0, self.q_es[j] - self.config.MES_TOTAL_CPU_CAPACITY * self.config.TAU / 10.0)
