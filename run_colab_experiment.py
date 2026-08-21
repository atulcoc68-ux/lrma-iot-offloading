import os
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from src.lrma_trainer import LRMATrainer
    from src.lrma_candidates import select_best_candidate
except ModuleNotFoundError:
    from lrma_trainer import LRMATrainer
    from lrma_candidates import select_best_candidate

# ==========================================
# 1. CONFIGURATION
# ==========================================
class EnvConfig:
    NUM_ED = 25
    NUM_MES = 5
    V = 20.0  # Lyapunov penalty factor
    SEQ_LENGTH = 10
    BATCH_SIZE = 64
    UPDATE_INTERVAL = 64
    DELTA_RESET = 50
    XI_SOFT = 0.01
    LR_ACTOR = 0.001
    LR_CRITIC = 0.001
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    @staticmethod
    def find_pod_file():
        colab_pods = glob.glob("/content/openb_pod_list_*.csv")
        if colab_pods:
            return colab_pods[0]
        local_pods = glob.glob(os.path.join(EnvConfig.BASE_DIR, "data", "Original dataset", "openb_pod_list_*.csv"))
        if local_pods:
            return local_pods[0]
        return "/content/openb_pod_list_gpuspec33.csv"

    @staticmethod
    def find_node_file():
        colab_nodes = glob.glob("/content/openb_node_list_gpu_node.csv")
        if colab_nodes:
            return colab_nodes[0]
        local_nodes = glob.glob(os.path.join(EnvConfig.BASE_DIR, "data", "Original dataset", "openb_node_list_gpu_node.csv"))
        if local_nodes:
            return local_nodes[0]
        return "/content/openb_node_list_gpu_node.csv"

# ==========================================
# 2. DATA LOADER & TASK
# ==========================================
class LRMATask:
    def __init__(self, name, start_slot, cpu_milli, gpu_milli, gpu_spec, duration):
        self.name = name
        self.start_slot = start_slot
        self.C = float(cpu_milli) / 1000.0 if pd.notna(cpu_milli) else 1.0
        self.G = float(gpu_milli) / 1000.0 if pd.notna(gpu_milli) else 0.5
        self.R = str(gpu_spec) if pd.notna(gpu_spec) and str(gpu_spec).strip() != '' else 'MISC'
        self.duration = float(duration) if pd.notna(duration) else 1.0
        self.size = max(0.1, (self.C + self.G) * 0.5)

class AlibabaWorkloadLoader:
    def __init__(self, pods_file, nodes_file):
        self.pods_df = pd.read_csv(pods_file)
        self.nodes_df = pd.read_csv(nodes_file)
        self._preprocess()

    def _preprocess(self):
        self.pods_df['gpu_spec'] = self.pods_df['gpu_spec'].fillna('MISC')
        min_time = self.pods_df['creation_time'].min()
        self.pods_df['start_slot'] = self.pods_df['creation_time'] - min_time
        self.pods_df['duration'] = self.pods_df['deletion_time'] - self.pods_df['creation_time']

# ==========================================
# 3. LSTM PREDICTOR
# ==========================================
class WorkloadPredictor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2):
        super(WorkloadPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class WorkloadDataset(Dataset):
    def __init__(self, data, seq_length=10):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_length].view(-1, 1), self.data[idx+self.seq_length]

def train_predictor(model, data, epochs=5, seq_length=10, lr=0.001):
    if len(data) <= seq_length: return
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = WorkloadDataset(data, seq_length)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred.squeeze(), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"LSTM Predictor Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")

def get_future_workload_estimate(model, history_seq):
    model.eval()
    with torch.no_grad():
        if len(history_seq) < 10:
            history_seq = [0.0] * (10 - len(history_seq)) + list(history_seq)
        x = torch.FloatTensor(history_seq[-10:]).view(1, -1, 1)
        pred = model(x)
        return float(pred.item())

# ==========================================
# 4. ENVIRONMENT & REWARD
# ==========================================
class MHFQ:
    def __init__(self, num_mes=5):
        self.num_mes = num_mes
        self.queues = {i: {} for i in range(num_mes)}

    def enqueue(self, mes_idx, task):
        gpu_type = task.R
        if gpu_type not in self.queues[mes_idx]:
            self.queues[mes_idx][gpu_type] = []
        self.queues[mes_idx][gpu_type].append(task)

    def get_queue_length(self, mes_idx, gpu_type):
        if gpu_type in self.queues[mes_idx]:
            return len(self.queues[mes_idx][gpu_type])
        return 0

class WirelessModel:
    def __init__(self, bandwidth_mhz=20.0, power_dbm=23.0, noise_dbm=-100.0):
        self.bandwidth = bandwidth_mhz * 1e6
        self.power = 10 ** (power_dbm / 10.0) / 1000.0
        self.noise = 10 ** (noise_dbm / 10.0) / 1000.0

    def calculate_rate(self):
        snr = self.power / self.noise
        rate_bps = self.bandwidth * np.log2(1 + snr)
        return rate_bps / 1e6

class LRMARewardCalculator:
    def __init__(self, V_penalty=20.0):
        self.V = V_penalty

    def calculate_drift(self, q_before, q_after):
        L_before = 0.5 * np.sum(q_before ** 2)
        L_after = 0.5 * np.sum(q_after ** 2)
        return L_after - L_before

    def calculate_reward(self, drift, energy, delay):
        performance_utility = - (0.5 * energy + 0.5 * delay)
        return self.V * performance_utility - drift

class LRMA_Environment:
    def __init__(self, loader, predictor, config):
        self.loader = loader
        self.predictor = predictor
        self.config = config
        self.mhfq = MHFQ(config.NUM_MES)
        self.wireless = WirelessModel()
        self.reward_calc = LRMARewardCalculator(V_penalty=config.V)
        self.history_workload = []

    def get_ed_state(self, ed_idx, task):
        q_local = task.size
        q_mes_sum = sum([self.mhfq.get_queue_length(i, task.R) for i in range(self.config.NUM_MES)])
        task_features = [task.C, task.G, (1.0 if task.R != 'MISC' else 0.0), task.size]
        pending_count = 1.0
        pending_size = task.size
        history_seq = self.history_workload[-10:] if len(self.history_workload) >= 10 else [0.0] * 10
        beta_hat = get_future_workload_estimate(self.predictor, history_seq)
        return np.array([task.size, task.C, task.G, (1.0 if task.R != 'MISC' else 0.0), pending_count, pending_size, q_local, q_mes_sum, beta_hat], dtype=np.float32)

    def get_cloud_state(self, task):
        offloaded_count = 1.0
        q_mes_list = [self.mhfq.get_queue_length(i, task.R) for i in range(self.config.NUM_MES)]
        history_seq = self.history_workload[-10:] if len(self.history_workload) >= 10 else [0.0] * 10
        beta_hat = get_future_workload_estimate(self.predictor, history_seq)
        return np.array([task.size, task.C, task.G, (1.0 if task.R != 'MISC' else 0.0), offloaded_count] + q_mes_list + [beta_hat], dtype=np.float32)

    def step(self, task, ed_action, cloud_action):
        if ed_action == 0:
            energy, delay = 0.5 * task.size, 1.0 * task.size
        else:
            mes_idx = cloud_action
            self.mhfq.enqueue(mes_idx, task)
            trans_rate = self.wireless.calculate_rate()
            trans_time = task.size / (trans_rate / 8.0)
            queue_len = self.mhfq.get_queue_length(mes_idx, task.R)
            energy, delay = 0.2 * task.size, trans_time + (queue_len / 10.0)
        return energy, delay

# ==========================================
# 5. MAIN SIMULATION EXPERIMENT
# ==========================================
def main(tiny_mode=False):
    pods_file = EnvConfig.find_pod_file()
    nodes_file = EnvConfig.find_node_file()
    
    print("=" * 60)
    print("LRMA Paper-Faithful CTDE Experiment")
    print("=" * 60)
    print(f"Loading data from:\n - Pods: {pods_file}\n - Nodes: {nodes_file}")

    loader = AlibabaWorkloadLoader(pods_file, nodes_file)
    print(f"Loaded {len(loader.pods_df)} tasks and {len(loader.nodes_df)} GPU nodes.")

    print("\n--- Pre-training LSTM Workload Predictor ---")
    raw_workload_data = loader.pods_df['cpu_milli'].fillna(1000.0).values / 1000.0
    predictor = WorkloadPredictor(input_dim=1, hidden_dim=64, num_layers=2)
    train_predictor(predictor, raw_workload_data, epochs=5, seq_length=EnvConfig.SEQ_LENGTH)

    num_ed = 2 if tiny_mode else EnvConfig.NUM_ED
    num_mes = EnvConfig.NUM_MES
    trainer = LRMATrainer(num_ed=num_ed, num_mes=num_mes, lr_actor=EnvConfig.LR_ACTOR, lr_critic=EnvConfig.LR_CRITIC)
    reward_calc = LRMARewardCalculator(V_penalty=EnvConfig.V)
    env = LRMA_Environment(loader, predictor, EnvConfig)

    tasks_df = loader.pods_df.head(10) if tiny_mode else loader.pods_df
    print(f"\n--- Running LRMA Paper-Faithful Multi-Agent Simulation ({len(tasks_df)} tasks) ---")

    lrma_history = []
    episode_reward = 0
    start_time = time.time()

    for i, row in tasks_df.iterrows():
        task = LRMATask(row['name'], row['start_slot'], row['cpu_milli'],
                        row['gpu_milli'], row['gpu_spec'], row['duration'])

        # State construction
        ed_states = [env.get_ed_state(e, task) for e in range(num_ed)]
        cloud_state = env.get_cloud_state(task)
        joint_state = np.concatenate([np.array(ed_states).flatten(), cloud_state])

        # Candidate generation & Argmax selection
        ed_candidates, _ = trainer.select_ed_action(0, ed_states[0], P=5)
        ed_action, _ = select_best_candidate(ed_candidates, lambda a: 1.0 if a == 0 else 0.5)

        if ed_action == 1:
            cloud_candidates, _ = trainer.select_cloud_action(cloud_state, P=5)
            cloud_action, _ = select_best_candidate(cloud_candidates, lambda a: -0.1 * a)
        else:
            cloud_action = 0

        # Joint action construction
        all_ed_actions = [ed_action] + [0] * (num_ed - 1)
        joint_action = trainer.construct_joint_action_representation(all_ed_actions, cloud_action)

        # Environment Step
        q_before = np.array([env.mhfq.get_queue_length(j, task.R) for j in range(num_mes)])
        energy, delay = env.step(task, ed_action, cloud_action)
        q_after = np.array([env.mhfq.get_queue_length(j, task.R) for j in range(num_mes)])

        drift = reward_calc.calculate_drift(q_before, q_after)
        reward = reward_calc.calculate_reward(drift, energy, delay)

        # Next state
        next_ed_states = [env.get_ed_state(e, task) for e in range(num_ed)]
        next_cloud_state = env.get_cloud_state(task)
        next_joint_state = np.concatenate([np.array(next_ed_states).flatten(), next_cloud_state])

        # Persistent experience replay insertion
        trainer.replay_buffer_ed.add(joint_state, joint_action, reward, next_joint_state, False)

        episode_reward += reward
        env.history_workload.append(task.C)

        # Periodic CTDE minibatch update & Parameter reset every 50 slots
        if (i + 1) % EnvConfig.UPDATE_INTERVAL == 0:
            a_loss, c_loss = trainer.train_step(batch_size=EnvConfig.BATCH_SIZE, xi_soft=EnvConfig.XI_SOFT)
            lrma_history.append(episode_reward)
            
            if (i + 1) % EnvConfig.DELTA_RESET == 0:
                trainer.reset_primary_parameters()
                print(f"Slot {i+1}: Reset primary network parameters.")
                
            if (i + 1) % (EnvConfig.UPDATE_INTERVAL * 10) == 0:
                print(f"Task {i+1}/{len(tasks_df)} | Interval Reward: {episode_reward:.2f} | Actor Loss: {a_loss:.4f} | Critic Loss: {c_loss:.4f}")
            episode_reward = 0

    elapsed = time.time() - start_time
    print(f"LRMA Training Complete in {elapsed:.2f} seconds.")

    out_dir = "/content" if os.path.exists("/content") else EnvConfig.BASE_DIR
    torch.save(trainer.critic.state_dict(), os.path.join(out_dir, "lrma_critic_ctde.pth"))
    print("Saved paper-faithful CTDE model checkpoint.")

if __name__ == "__main__":
    main(tiny_mode=True)  # Lightweight local sanity mode
