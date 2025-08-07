import torch
import torch.nn as nn

import numpy as np

class BasicRegressionBlock(nn.Module):
    def __init__(self,
                 n_hidden_dims: int,
                 n_output_channels: int,
                 n_layers: int = 4):
        super().__init__()

        exp = 1/(n_layers - 1)
        r = np.power(n_output_channels / n_hidden_dims, exp)

        output_dims = []
        for idx in range(n_layers):
            output_dims.append(int(np.round(n_hidden_dims * np.power(r, idx))))

        self.linear_layers = nn.Sequential()

        for idx in range(output_dims-1):
            self.linear_layers.append(nn.Linear(in_features=output_dims[idx], out_features=output_dims[idx+1]))

        self.nl = nn.ReLU()

    def forward(self, x):

        x_hat = torch.permute(x, dims=(0,2,1))

        for linear_layer in self.linear_layers:
            x_hat = linear_layer(x_hat)
            x_hat = self.nl(x_hat)

        x = torch.permute(x_hat, dims=(0,2,1))

        return x