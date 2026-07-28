# LRMA: Multi-Agent DRL-Based IoT Task Offloading

Implementation of the LRMA framework (LSTM + Parameter Reset + Multi-Agent DRL) with MHFQ queuing and Lyapunov optimization.

## Team Work Division
- **Member 1 (`src/environment.py`)**: IoT environment, channel model, MHFQ queues.
- **Member 2 (`src/lyapunov.py`, `src/lstm_model.py`)**: Virtual queue drift tracking, reward calculation, LSTM prediction network.
- **Member 3 (`src/agents.py`, `src/replay_buffer.py`, `train.py`)**: ED/Cloud Actor networks, parameter reset mechanism, main training loop.
