"""Node-shared forecasters and dueling Double-DQN."""

import torch
from torch import nn


class Forecaster(nn.Module):
    def __init__(self, kind="tgcn", hidden=32):
        super().__init__()
        self.kind = kind
        self.hidden = hidden
        if kind == "tgcn":
            self.gates = nn.Linear(hidden + 1, 2 * hidden)
            self.candidate = nn.Linear(hidden + 1, hidden)
        elif kind in ("gru", "lstm"):
            cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = cls(1, hidden, batch_first=True)
        elif kind == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(12, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
            )
        else:
            raise ValueError(kind)
        self.output = nn.Linear(hidden, 3)

    def aggregate(self, z, a):
        b, n, f = z.shape
        z = z.permute(1, 0, 2).reshape(n, b * f)
        z = torch.sparse.mm(a, z) if a.is_sparse else a @ z
        return z.reshape(n, b, f).permute(1, 0, 2)

    def forward(self, x, a):
        b, t, n = x.shape
        if self.kind == "tgcn":
            h = x.new_zeros(b, n, self.hidden)
            for k in range(t):
                xt = x[:, k, :, None]
                r, u = torch.sigmoid(
                    self.gates(self.aggregate(torch.cat([xt, h], -1), a))
                ).chunk(2, -1)
                c = torch.tanh(
                    self.candidate(self.aggregate(torch.cat([xt, r * h], -1), a))
                )
                h = u * h + (1 - u) * c
        elif self.kind in ("gru", "lstm"):
            h = self.rnn(x.transpose(1, 2).reshape(b * n, t, 1))[0][:, -1].reshape(
                b, n, self.hidden
            )
        else:
            h = self.mlp(x.transpose(1, 2))
        # Same persistence residual for all learned baselines.
        return self.output(h).transpose(1, 2) + x[:, -1:, :]


class DuelingQ(nn.Module):
    def __init__(self, inputs, actions=4, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, actions)

    def forward(self, x):
        h = self.body(x)
        a = self.advantage(h)
        return self.value(h) + a - a.mean(-1, keepdim=True)
