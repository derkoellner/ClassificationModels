import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class DiagonalLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_features))

        if bias:
            self.bias = nn.Parameter(torch.randn(in_features))
        
    def forward(self, x):

        if self.bias:
            return x * self.weight + self.bias
        else:
            return x * self.weight

class BiasLayer(nn.Module):
    def __init__(
            self,
            n_inputs: int):
        super(BiasLayer, self).__init__()
        self.bias = nn.Parameter(torch.randn(n_inputs)) # TODO ensure its learnable
    
    def forward(self, x):
        return self.bias
    
class BaseStackedAutoEncoder(nn.Module):
    def get_number_layers(self,
                          n_channels: int,
                          n_joints: int,
                          decrease: int):
        
        output_size = 2 * n_joints
        n_layers = max(1, (n_channels - output_size + decrease - 1) // decrease + 1)
        layer_inputs = np.maximum(n_channels - decrease * np.arange(n_layers), output_size)
        return n_layers, layer_inputs.astype(int)
    
    def get_activation_function(self,
                                activation_function: str,
                                dim: int=1):
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softmax": nn.Softmax(dim=dim),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "gelu": nn.GELU()
        }
        return activations.get(activation_function.lower(), nn.ReLU())
    
class _ResidualBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 3,
            dilation: int = 1,
            dropout: float = 0.5
    ):
        super().__init__()

        self.padding = (kernel_size-1)*dilation

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels,
                      kernel_size=kernel_size,
                      dilation=dilation,
                    #   padding=(padding, 0),
                    #   padding_mode='zeros'),
            ),
            nn.InstanceNorm1d(in_channels, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels, in_channels,
                      kernel_size=kernel_size,
                      dilation=dilation,
                    #   padding=(padding, 0),
                    #   padding_mode='zeros'),
            ),
            nn.InstanceNorm1d(in_channels, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.residual_conv = nn.Conv1d(in_channels, in_channels,
                                       kernel_size=1)
        
        self.nl = nn.ReLU()
        
    def forward(self, x):
        x_pad = F.pad(x, (self.padding, 0))
        x_conv = self.conv_1(x_pad)
        x_pad = F.pad(x_conv, (self.padding, 0))
        x_conv = self.conv_2(x_pad)

        x_residual = self.residual_conv(x)

        return self.nl(x_conv + x_residual)