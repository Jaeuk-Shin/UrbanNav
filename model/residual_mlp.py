import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.gelu = nn.GELU()

    def forward(self, x):
        # Residual connection: x + f(x)
        return self.gelu(x + self.block(x))


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=1024, num_blocks=3):
        super().__init__()
        
        # 1. Project input to hidden dimension
        self.initial_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. Stack of Residual Blocks
        self.res_stack = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_blocks)]
        )
        
        # 3. Final output head
        self.final_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.initial_projection(x)
        x = self.res_stack(x)
        return self.final_head(x)