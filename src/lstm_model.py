import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig


class WorkloadPredictor(nn.Module):
    r"""
    LSTM model to predict future task arrival states \beta^t (Paper Section V-C.1, Eq 41-42).
    Learns from historical arrival state sequences \widetilde{T} = {\widetilde{T}^{t-l}, ..., \widetilde{T}^t}
    to predict future arrival state vector \beta^t (number, type, resource requirements).
    """
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=4, num_layers=2):
        super(WorkloadPredictor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Paper Eq. (41): Long short-term memory forward pass
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


class WorkloadSequenceDataset(Dataset):
    r"""
    Dataset wrapping historical task arrival sequences \widetilde{T}.
    """
    def __init__(self, state_sequences, seq_length=EnvConfig.SEQ_LENGTH):
        self.seq_length = seq_length
        self.data = torch.FloatTensor(np.array(state_sequences))

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        x_seq = self.data[idx : idx + self.seq_length]
        y_target = self.data[idx + self.seq_length]
        return x_seq, y_target


def train_predictor(model, historical_states, epochs=5, seq_length=EnvConfig.SEQ_LENGTH, lr=EnvConfig.LR_CRITIC):
    r"""
    Trains the LSTM predictor using Adam optimizer and MSE loss (Paper Eq 42).
    Loss_{lstm} = \frac{1}{N+1} \sum_{i=1}^{N+1} (\widetilde{T}^t - \beta^t)^2
    """
    if len(historical_states) <= seq_length + 10:
        return
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Paper Eq. (42)
    
    dataset = WorkloadSequenceDataset(historical_states, seq_length)
    loader = DataLoader(dataset, batch_size=min(EnvConfig.BATCH_SIZE, len(dataset)), shuffle=True)

    for epoch in range(epochs):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            beta_pred = model(x_batch)
            loss = criterion(beta_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(1, len(loader))
        if (epoch + 1) == epochs or epoch == 0:
            print(f"LSTM Workload Predictor Epoch {epoch+1}/{epochs} | Loss (MSE): {avg_loss:.6f}")


def get_future_workload_estimate(model, history_seq):
    """
    Helper to predict future arrival state vector \beta^t given past arrival state sequence.
    """
    model.eval()
    with torch.no_grad():
        history_arr = np.array(history_seq, dtype=np.float32)
        if len(history_arr) < EnvConfig.SEQ_LENGTH:
            pad_size = EnvConfig.SEQ_LENGTH - len(history_arr)
            pad = np.zeros((pad_size, history_arr.shape[1] if history_arr.ndim > 1 else 4), dtype=np.float32)
            history_arr = np.vstack([pad, history_arr]) if history_arr.ndim > 1 else pad
        
        if history_arr.ndim == 1:
            history_arr = np.tile(history_arr, (EnvConfig.SEQ_LENGTH, 1))

        x_tensor = torch.FloatTensor(history_arr[-EnvConfig.SEQ_LENGTH:]).unsqueeze(0)
        beta_hat = model(x_tensor).squeeze(0).numpy()
        return beta_hat
