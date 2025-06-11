import torch
import torch.nn as nn

from .Encoders.Stacked import StackedEncoder
from .Decoders.Stacked import StackedDecoder

from .Encoders.CNN import CNN_Spec, CNN_Temp
# from .Decoders.CNN import CNN_Parallel, CNN_Series
    
class StackedAE(nn.Module):
    def __init__(
            self,
            n_channels: int,
            hidden_size: int = 2,
            decrease: int = 1,
            bias: bool = True,
            activation_function: str = 'ReLU',
            output_activation_function: str = 'sigmoid'):
        super().__init__()

        self.Encoder = StackedEncoder(n_channels=n_channels, hidden_size=hidden_size, decrease=decrease, bias=bias, activation_function=activation_function, output_activation_function=output_activation_function)
        self.Decoder = StackedDecoder(n_channels=n_channels, hidden_size=hidden_size, decrease=decrease, bias=bias, activation_function=activation_function)

    def forward(self, x, training_iteration=0):

        if self.training and training_iteration!=0:
            x, in_features = self.Encoder(x, training_iteration)
            x, out_features = self.Decoder(x, training_iteration)

            return x, in_features, out_features

        x = self.Encoder(x, training_iteration)
        x = self.Decoder(x, training_iteration)

        return x
    
class Combined_CNN(nn.Module):
    def __init__(self,
                spec_shape: tuple,
                n_time_samples: int,
                n_channels: int,
                n_temporal_filters: int = 40,
                temporal_filter_length: float = 0.5,
                dropout: float = 0.5,
                hidden_size: int = 2):
                # parallel: bool = False):
        super().__init__()

        self.spectral_encoder = CNN_Spec(spec_shape=spec_shape, n_time_samples=n_time_samples, n_channels=n_channels, n_temporal_filters=n_temporal_filters, temporal_filter_length=temporal_filter_length, dropout=dropout)
        self.time_encoder = CNN_Temp(n_time_samples=n_time_samples, n_channels=n_channels, n_temporal_filters=n_temporal_filters, dropout=dropout)

        self.linear1 = nn.Linear(in_features=2*n_temporal_filters, out_features=n_temporal_filters)
        self.nl = nn.ReLU()
        self.linear2 = nn.Linear(in_features=n_temporal_filters, out_features=hidden_size)

        # if parallel:
        #     self.decoder = CNN_Parallel(n_time_samples=n_time_samples, n_channels=n_channels, dropout=dropout)
        # else:
        #     self.decoder = CNN_Series(n_time_samples=n_time_samples, n_channels=n_channels, dropout=dropout)

    def forward(self, x_f, x_t):
        x_f = self.spectral_encoder(x_f)
        x_t = self.time_encoder(x_t)

        x_f = torch.permute(x_f, dims=(0,2,1))
        x_t = torch.permute(x_t, dims=(0,2,1))

        x_hat = torch.concat((x_f, x_t), dim=2)

        x_hat = self.linear1(x_hat)
        x_hat = self.nl(x_hat)
        x_hat = self.linear2(x_hat)

        x_hat = torch.permute(x_hat, dims=(0,2,1))

        # x = self.decoder(x_hat)

        return x_hat