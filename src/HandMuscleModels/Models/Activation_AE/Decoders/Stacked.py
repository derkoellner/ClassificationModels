import numpy as np
import torch.nn as nn
from ..Helper_Modules import BaseStackedAutoEncoder

class StackedDecoder(BaseStackedAutoEncoder):
    def __init__(
            self,
            n_channels: int,
            n_joints: int = 1,
            decrease: int = 1,
            bias: bool = True,
            activation_function: str = 'ReLU'):
        super().__init__()
        self.Decoder_Layers = nn.ModuleList()
        self.activation_function = self.get_activation_function(activation_function=activation_function)
        n_layers, layer_inputs = self.get_number_layers(n_channels=n_channels, n_joints=n_joints, decrease=decrease)
        layer_inputs = np.flip(layer_inputs)
        for layer in range(n_layers-1):
            layer_input = layer_inputs[layer]
            layer_output = layer_inputs[layer+1]
            self.Decoder_Layers.append(nn.Linear(in_features=layer_input, out_features=layer_output, bias=bias))
        self.n_layers = n_layers-1

    def forward(self, x, training_iteration=0):

        if self.training:
            for i in reversed(range(training_iteration+1)):
                x = self.Decoder_Layers[-(i+1)](x)
                x = self.activation_function(x)

                if training_iteration == 0:
                    return x
                elif i == training_iteration:
                    output_features = x

            return x, output_features

        elif not self.training:
            for layer in self.Decoder_Layers:
                x = layer(x)
                x = self.activation_function(x)
        return x