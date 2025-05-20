import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
    
class CNN_Spec(nn.Module):
    def __init__(self,
                 spec_shape: tuple,
                 t_samples: int,
                 n_channels: int,
                 n_temporal_filters: int = 40,
                 temporal_filter_length: float = 0.5,
                 dropout: float = 0.5,
                 combine_with_temporal: bool = True):
        super().__init__()

        n_freq_samples = spec_shape[0]
        n_time_samples = spec_shape[1]

        if combine_with_temporal:
            downsample_size = n_temporal_filters
        else:
            downsample_size = 2
        
        temporal_filter_length = int(n_time_samples * temporal_filter_length)
        temporal_filter_length = temporal_filter_length - 1 if temporal_filter_length % 2 == 0 else temporal_filter_length

        self.temp_freq_conv = nn.Conv2d(n_channels, n_temporal_filters*n_channels,
                                        kernel_size=(n_freq_samples, temporal_filter_length),
                                        padding=(0, temporal_filter_length // 2),
                                        bias=False,
                                        groups=n_channels)
        self.intermediate_bn1 = nn.BatchNorm2d(n_temporal_filters*n_channels)
        self.spatial_conv = nn.Conv2d(n_temporal_filters*n_channels, n_temporal_filters,
                                      kernel_size=(1,1),
                                      bias=False)
        self.intermediate_bn2 = nn.BatchNorm2d(n_temporal_filters)
        self.spatial_conv_downsampling = nn.Conv2d(n_temporal_filters, downsample_size,
                                kernel_size=(1,1),
                                bias=False)
        self.bn = nn.BatchNorm2d(downsample_size)

        self.upsample = nn.Upsample(size=t_samples, mode='linear', align_corners=True)
        self.cnn = nn.Conv1d(in_channels=downsample_size, out_channels=downsample_size, kernel_size=3, padding=1)
        self.nl = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.temp_freq_conv(x)
        x = self.intermediate_bn1(x)
        x = self.spatial_conv(x)
        x = self.intermediate_bn2(x)
        x = self.spatial_conv_downsampling(x)
        x = self.bn(x)
        x = torch.squeeze(x, 2)
        x = self.nl(x)
        x = self.upsample(x)
        x = self.cnn(x)
        x = self.nl(x)
        x = self.dropout(x)

        return x
    
class CNN_Temp(nn.Module):
    def __init__(self,
                 n_time_samples: int,
                 n_channels: int,
                 n_temporal_filters: int = 20,
                 dropout: float = 0.5,
                 combine_with_spectrogram: bool = True):
        super().__init__()

        if combine_with_spectrogram:
            downsample_size = n_temporal_filters
        else:
            downsample_size = 2

        temporal_filter_length = int(n_time_samples * 0.1)
        temporal_filter_length = temporal_filter_length - 1 if temporal_filter_length % 2 == 0 else temporal_filter_length

        self.stretch_input = Rearrange("b c t -> b 1 c t")

        self.temporal_conv = nn.Conv2d(1, n_temporal_filters,
                                       kernel_size=(1,temporal_filter_length),
                                       padding=(0,temporal_filter_length // 2),
                                       bias=False)
        
        self.bn1 = nn.BatchNorm2d(n_temporal_filters)
        
        self.spatial_conv = nn.Conv2d(n_temporal_filters, n_temporal_filters,
                                      kernel_size=(n_channels,1),
                                      bias=False)

        self.bn2 = nn.BatchNorm2d(n_temporal_filters)

        self.dropout = nn.Dropout(dropout)

        self.nl = nn.ELU()

        self.linear = nn.Linear(in_features=n_temporal_filters, out_features=downsample_size)

    def forward(self, x):
        x = self.stretch_input(x)
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = torch.squeeze(x, dim=2)
        x = self.nl(x)
        x = torch.permute(x, dims=(0,2,1))
        x = self.linear(x)
        x = self.nl(x)
        x = torch.permute(x, dims=(0,2,1))

        return x