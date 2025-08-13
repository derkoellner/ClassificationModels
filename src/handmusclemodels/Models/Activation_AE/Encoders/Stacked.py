import torch.nn as nn
from ..Helper_Modules import BaseStackedAutoEncoder

class StackedEncoder(BaseStackedAutoEncoder):
    def __init__(
            self,
            n_channels: int,
            hidden_size: int = 2,
            decrease: int = 1,
            bias: bool = True,
            activation_function: str = 'ReLU',
            output_activation_function: str = 'sigmoid'):
        super().__init__()
        self.Encoder_Layers = nn.ModuleList()
        self.activation_function = self.get_activation_function(activation_function=activation_function)
        self.output_activation_function = self.get_activation_function(activation_function=output_activation_function)
        n_layers, layer_inputs = self.get_number_layers(n_channels=n_channels, hidden_size=hidden_size, decrease=decrease)
        for layer in range(n_layers-1):
            layer_input = layer_inputs[layer]
            layer_output = layer_inputs[layer+1]
            self.Encoder_Layers.append(nn.Linear(in_features=layer_input, out_features=layer_output, bias=bias))
        self.n_layers = n_layers-1

    def forward(self, x, training_iteration=0):

        if self.training:
            for i in range(training_iteration+1):
                x = self.Encoder_Layers[i](x)
                x = self.output_activation_function(x) if i == len(self.Encoder_Layers) - 1 else self.activation_function(x)

                if training_iteration == 0:
                    return x
                elif i == training_iteration-1:
                    input_features = x

            return x, input_features

        elif not self.training:
            for i, layer in enumerate(self.Encoder_Layers):
                x = layer(x)
                x = self.output_activation_function(x) if i == len(self.Encoder_Layers) - 1 else self.activation_function(x)
        return x