import torch.nn as nn
from ..Helper_Modules import _ResidualBlock

import numpy as np

class TCN_Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_time_samples: int,
            kernel_size: int = 3,
            n_residual_blocks: int = 0,
            dropout: float = 0.5
    ):
        super().__init__()

        self.tcn = nn.Sequential()

        if n_residual_blocks == 0:
            inbetween = (n_time_samples - 2 + kernel_size) / (kernel_size - 1)
            n_residual_blocks = int(np.ceil(np.log2(inbetween)))

        for i in range(n_residual_blocks):
            self.tcn.append(
                _ResidualBlock(
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                    dilation=2**i
                )
            )

        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, 2,
                    kernel_size=1),
            nn.InstanceNorm1d(2, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout))
        
    def forward(self, x):
        x = self.tcn(x)
        x = self.downsample(x)

        return x
