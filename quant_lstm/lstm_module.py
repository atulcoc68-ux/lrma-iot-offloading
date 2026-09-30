"""
lstm_module.py
==============
Verbatim extract of LSTM_Cell and LSTM from
  multitask_multilevel/memory.py  (lines 15-54).

The class bodies are UNCHANGED.  Only module-level imports are added so
this file can be imported without triggering gloable_variation.py (which
reads CSVs from hard-coded Windows paths).

LSTMWrapper is a NEW class added in quant_lstm/ that:
  - replaces torch.randn h/c init with torch.zeros (deterministic, ONNX-safe)
  - supports any batch size
  - is used for ALL variants (training, FP32 eval, dynamic PTQ,
    static PTQ, and ONNX export) to ensure apples-to-apples comparison
"""

import torch
import torch.nn as nn


# ── Verbatim extract — memory.py L15-34 ─────────────────────────────────────

class LSTM_Cell(nn.Module):
    def __init__(self, in_dim , hidden_dim):
        super(LSTM_Cell,self).__init__()
        self.ix_linear = nn.Linear(in_dim,hidden_dim)
        self.ih_linear = nn.Linear(hidden_dim,hidden_dim)
        self.fx_linear = nn.Linear(in_dim, hidden_dim)
        self.fh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.ox_linear = nn.Linear(in_dim, hidden_dim)
        self.oh_linear = nn.Linear(hidden_dim, hidden_dim)
        self.cx_linear = nn.Linear(in_dim, hidden_dim)
        self.ch_linear = nn.Linear(hidden_dim, hidden_dim)
    def forward(self,x , h_1,c_1):
        i = torch.sigmoid(self.ix_linear(x)+self.ih_linear(h_1))
        f = torch.sigmoid(self.fx_linear(x)+self.fh_linear(h_1))
        o = torch.sigmoid(self.ox_linear(x) + self.oh_linear(h_1))
        c_ = torch.tanh(self.cx_linear(x) + self.ch_linear(h_1))
        c = i * c_ + f *c_1
        h = o * torch.tanh(c)
        h = torch.sigmoid(h)
        return  h , c


# ── Verbatim extract — memory.py L36-54 ─────────────────────────────────────

class LSTM(nn.Module):
    def __init__(self, in_dim , hidden_dim):
        super(LSTM,self).__init__()
        self.hidden_dim =hidden_dim
        self.lstm_cell = LSTM_Cell(in_dim , hidden_dim)
    def forward(self,x):
        '''
        x = [seq_lens, batch_size, in_dim]
        '''
        outs=[]
        h,c =None,None
        for seq_x in x:
            #seq_x : [batch,in_dim]
            if h is None: h = torch.randn(1,self.hidden_dim)
            if c is None: c = torch.randn(1,self.hidden_dim)
            h,c = self.lstm_cell(seq_x,h,c)
            outs.append(torch.unsqueeze(h,0))
        outs = torch.cat(outs)
        return outs,h


# ── LSTMWrapper — NEW in quant_lstm/ ────────────────────────────────────────

class LSTMWrapper(nn.Module):
    """
    Wraps LSTM with DETERMINISTIC zero h/c initialization.

    Why needed
    ----------
    The original LSTM.forward() calls torch.randn(1, hidden_dim) for h and c
    on the first timestep (memory.py L49-50).  This makes:
      • repeated forward passes non-reproducible (different random h/c each time),
      • ONNX tracing non-deterministic (randn is a non-const op),
      • latency benchmarks noisy,
      • and FP32 vs quantized comparisons unfair.

    This wrapper bypasses the None-check by calling lstm_cell directly with
    torch.zeros, and supports arbitrary batch sizes (the original is hardcoded
    to batch=1).

    Used for ALL variants (training, FP32 eval, dynamic PTQ, static PTQ,
    and ONNX export) to ensure identical initialisation across comparisons.

    Output range
    ------------
    LSTM_Cell.forward() applies torch.sigmoid(h) as its last op (memory.py L33).
    All values of h (and therefore predict_L) are in (0, 1).
    BCELoss is mathematically valid on the prediction side.
    Targets built from user_predict() may exceed 1.0 (see dataset.py for details
    and the clamping strategy).
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = LSTM(in_dim, hidden_dim)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        """
        Args
        ----
        x : (seq_len, batch, in_dim)   — same as LSTM.forward()

        Returns
        -------
        outs : (seq_len, batch, hidden_dim)
        h    : (batch, hidden_dim)     — last hidden state  ∈ (0, 1)
        """
        batch = x.shape[1]
        h = torch.zeros(batch, self.hidden_dim, dtype=x.dtype, device=x.device)
        c = torch.zeros(batch, self.hidden_dim, dtype=x.dtype, device=x.device)
        outs = []
        for seq_x in x:           # seq_x : (batch, in_dim)
            h, c = self.lstm.lstm_cell(seq_x, h, c)
            outs.append(h.unsqueeze(0))
        outs = torch.cat(outs, dim=0)
        return outs, h
