import numpy as np

try:
    from src.config import EnvConfig
except ModuleNotFoundError:
    from config import EnvConfig


class WirelessModel:
    """
    Wireless Communication Model (Section III-B of IEEE TNSE Paper, Eq 6-10).
    Models signal propagation, channel gain, Shannon transmission capacity, and offloading delays.
    """
    def __init__(self, bandwidth=EnvConfig.BANDWIDTH, noise_power=EnvConfig.SIGMA_SQUARED,
                 trans_power=EnvConfig.TRANS_POWER, antenna_gain=EnvConfig.ANTENNA_GAIN,
                 carrier_freq=EnvConfig.CARRIER_FREQ, path_loss_exponent=EnvConfig.PATH_LOSS_EXP,
                 overhead=EnvConfig.TRANS_OVERHEAD):
        self.B = bandwidth             # B_i^t (Hz)
        self.sigma2 = noise_power      # \sigma^2 (-60 dBm = 1e-9 W)
        self.p_tx = trans_power        # P_i^{tran} (W)
        self.a_antenna = antenna_gain  # A_{antenna}
        self.f_carrier = carrier_freq  # f_{carrier} (Hz)
        self.loss_ple = path_loss_exponent # loss_{ple}
        self.overhead = overhead       # o_u(t) > 1

    def calculate_distance(self, ed_pos, bs_pos):
        """Calculates distance between ED i and BS j (Paper Eq. 7)."""
        # Paper Eq. (7): dist_{i,j}^t = \sqrt{(l_i^{ED,x}(t) - l_j^{BS,x})^2 + (l_i^{ED,y}(t) - l_j^{BS,y})^2}
        dx = ed_pos[0] - bs_pos[0]
        dy = ed_pos[1] - bs_pos[1]
        return max(1.0, np.sqrt(dx**2 + dy**2))

    def calculate_channel_gain(self, distance):
        """Calculates channel gain h_{i,j}^t between ED i and BS j (Paper Eq. 6)."""
        # Paper Eq. (6): h_{i,j}^t = A_{antenna} * (3*10^8 / (4*pi*f_{carrier}*dist_{i,j}^t))^{loss_{ple}}
        c = 3.0 * 1e8
        gain = self.a_antenna * ((c / (4.0 * np.pi * self.f_carrier * distance)) ** self.loss_ple)
        return max(1e-6, gain)

    def calculate_rate(self, channel_gain=1e-3):
        """Calculates transmission rate v_{i,j}^t in bit/s (Paper Eq. 8)."""
        # Paper Eq. (8): v_{i,j}^t = B_i^t * log_2(1 + (P_i^{tran} * h_{i,j}^t) / \sigma^2)
        snr = (self.p_tx * channel_gain) / self.sigma2
        rate_bps = self.B * np.log2(1.0 + snr)
        return max(1e3, rate_bps)

    def calculate_transmission_delay(self, task_size_bits, rate_bps):
        """Calculates transmission delay t_{i,j}^{tran,t} in seconds (Paper Eq. 9)."""
        # Paper Eq. (9): t_{i,j}^{tran,t} = (o_u(t) * size_{i,k}^t) / v_{i,j}^t
        return (self.overhead * task_size_bits) / max(1e3, rate_bps)

    def calculate_offload_completion_entry_time(self, t_gen, trans_delay):
        """Calculates entry time into top queue to_{j,r}^1(T_{i,k}^t) (Paper Eq. 10)."""
        # Paper Eq. (10): to_{j,r}^1(T_{i,k}^t) = t_{i,j}^{tran,t} + t
        return trans_delay + t_gen
