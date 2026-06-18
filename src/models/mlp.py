import torch.nn as nn
import torch.nn.functional as F


class ResMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.15):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.norm = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + x


class ResMLPClassifier(nn.Module):
    def __init__(self, input_dim, dim=256, n_blocks=3, hidden_mult=2, dropout=0.15):
        super().__init__()
        self.input = nn.Sequential(nn.BatchNorm1d(input_dim), nn.Linear(input_dim, dim))
        self.blocks = nn.Sequential(
            *[
                ResMLPBlock(dim=dim, hidden_dim=dim * hidden_mult, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)


class GEGLUBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.norm = nn.BatchNorm1d(dim)
        self.fc = nn.Linear(dim, hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, gate = self.fc(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.dropout(x)
        x = self.proj(x)
        return residual + x


class GatedResMLPClassifier(nn.Module):
    def __init__(self, input_dim, dim=256, n_blocks=3, hidden_mult=2, dropout=0.2):
        super().__init__()
        self.input = nn.Sequential(nn.BatchNorm1d(input_dim), nn.Linear(input_dim, dim))
        self.blocks = nn.Sequential(
            *[
                GEGLUBlock(dim=dim, hidden_dim=dim * hidden_mult, dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.blocks(x)
        return self.head(x).squeeze(1)
