import torch
import torch.nn as nn

class GRU_Decoder(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_time_samples: int,
            hidden_size: int = 0,
            hidden_gru_size: int = 0,
            dropout: float = 0.5):
        super().__init__()

        self.n_time_samples = n_time_samples
        if hidden_gru_size == 0:
            hidden_gru_size = n_channels

        self.upsample = nn.Sequential(
            nn.Conv1d(hidden_size,n_channels,
                      kernel_size=1),
            nn.InstanceNorm1d(n_channels, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(
            input_size=n_channels,
            hidden_size=hidden_gru_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0
        )

        self.reconstr = nn.Linear(hidden_gru_size, n_channels)

    def forward(self, x):
        x = self.upsample(x)

        x = torch.permute(x, dims=(0,2,1))

        x_reversed = torch.flip(x, dims=[1])

        output_reversed,_ = self.gru(x_reversed)

        output = torch.flip(output_reversed, dims=[1])

        z = self.reconstr(output)

        z = torch.permute(z, (0,2,1))

        return z
    
class CostumRNN_Decoder(nn.Module):
    def __init__(
            self,
            n_channels: int,
            n_time_samples: int,
            hidden_size: int = 2,
            dropout: float = 0.5):
        super().__init__()

        self.register_buffer('h_0', torch.zeros(n_channels))
        self.n_time_samples = n_time_samples

        self.upsample = nn.Sequential(
            nn.Conv1d(hidden_size,n_channels,
                      kernel_size=1),
            nn.InstanceNorm1d(n_channels, affine=True),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.reset_gate = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid()
        )

        self.update_gate = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.Sigmoid()
        )

        self.proposed_state = nn.Sequential(
            nn.Linear(n_channels, n_channels),
            nn.Tanh()
        )

        self.reconstr = nn.Linear(n_channels, n_channels)

    def forward(self, x):
        x = self.upsample(x)

        h_t_1 = self.h_0

        z = torch.zeros_like(x)

        for t in range(1,self.n_time_samples+1):

            x_t = x[:,:,-t]
            r_t = self.reset_gate(x_t + h_t_1)
            u_t = self.update_gate(x_t + h_t_1)
            h_t_hat = self.proposed_state(x_t + (r_t * h_t_1))
            h_t = (1-u_t) * h_t_1 + u_t * h_t_hat
            z[:,:,-t] = self.reconstr(h_t)

            h_t_1 = h_t

        return z