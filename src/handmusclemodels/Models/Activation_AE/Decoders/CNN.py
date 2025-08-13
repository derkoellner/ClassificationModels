import torch
import torch.nn as nn

class CNN_Parallel(nn.Module):
    def __init__(self,
                 n_time_samples: int,
                 n_channels: int,
                 dropout: float = 0.5,
                 hidden_size: int = 2):
        super().__init__()

        temporal_filter_length = int(n_time_samples * 0.1)
        temporal_filter_length = temporal_filter_length - 1 if temporal_filter_length % 2 == 0 else temporal_filter_length

        self.deconv_channels = nn.ConvTranspose1d(hidden_size, n_channels,
                                                  kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv_s_time = nn.Conv1d(n_channels, n_channels,
                                   kernel_size=temporal_filter_length,
                                   padding=temporal_filter_length // 2)
        self.bn2 = nn.BatchNorm1d(n_channels)
        
        temporal_filter_length = int(n_time_samples * 0.5)
        temporal_filter_length = temporal_filter_length - 1 if temporal_filter_length % 2 == 0 else temporal_filter_length

        self.conv_l_time = nn.Conv1d(n_channels, n_channels,
                                   kernel_size=temporal_filter_length,
                                   padding= (temporal_filter_length // 2) * 2,
                                   dilation=2)
        self.bn3 = nn.BatchNorm1d(n_channels)
        
        self.nl = nn.ReLU()

        self.linear_l = nn.Linear(in_features=n_channels, out_features=n_channels) # Added
        self.linear_s = nn.Linear(in_features=n_channels, out_features=n_channels) # Added
        self.linear_full = nn.Linear(in_features=n_channels, out_features=n_channels) # Added

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        x = self.deconv_channels(x)
        x = self.bn1(x)
        x = self.nl(x)

        x_st = self.conv_s_time(x)
        x_st = self.bn2(x_st)
        # x_st = self.nl(x_st)

        x_lt = self.conv_l_time(x)
        x_lt = self.bn3(x_lt)
        # x_lt = self.nl(x_lt)

        x_st = torch.permute(x_st, dims=(0,2,1)) # Added
        x_lt = torch.permute(x_lt, dims=(0,2,1)) # Added

        x_st = self.linear_s(x_st) # Added
        x_lt = self.linear_l(x_lt) # Added

        x = x_st + x_lt

        x = self.linear_full(x) # Added

        x = torch.permute(x, dims=(0,2,1)) # Added

        x = self.dropout(x)

        return x

class CNN_Series(nn.Module):
    def __init__(self,
                 n_time_samples: int,
                 n_channels: int,
                 dropout: float = 0.5,
                 hidden_size: int = 2):
        super().__init__()

        temporal_filter_length = int(n_time_samples * 0.1)
        temporal_filter_length = temporal_filter_length - 1 if temporal_filter_length % 2 == 0 else temporal_filter_length

        self.deconv_channels = nn.ConvTranspose1d(hidden_size, n_channels,
                                                  kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_channels)
        self.conv_s_time = nn.Conv1d(n_channels, n_channels,
                                   kernel_size=temporal_filter_length,
                                   padding=temporal_filter_length // 2)
        self.bn2 = nn.BatchNorm1d(n_channels)
        
        temporal_filter_length = int(n_time_samples * 0.5)
        temporal_filter_length = temporal_filter_length - 1 if temporal_filter_length % 2 == 0 else temporal_filter_length

        self.conv_l_time = nn.Conv1d(n_channels, n_channels,
                                   kernel_size=temporal_filter_length,
                                   padding= (temporal_filter_length // 2) * 2,
                                   dilation=2)
        self.bn3 = nn.BatchNorm1d(n_channels)
        
        self.nl = nn.ReLU()

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):

        x = self.deconv_channels(x)
        x = self.bn1(x)
        x = self.nl(x)

        x_st = self.conv_s_time(x)
        x_st = self.bn2(x_st)
        x_st = self.nl(x_st)
        x_st = x + x_st

        x_lt = self.conv_l_time(x_st)
        x_lt = self.bn3(x_lt)
        x_lt = self.nl(x_lt)
        x = x + x_st + x_lt

        x = self.dropout(x)

        return x