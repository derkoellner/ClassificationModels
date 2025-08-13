import numpy as np
import torch
import torch.nn as nn

from .DAE_functions import stretch, gaussian_interpolation, euler_encoding

class Stretch(nn.Module):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    def __init__(self, num_features, num_dims, alpha):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        self.alpha = alpha
        self.gamma = nn.Parameter(0.01*torch.ones(shape))
        self.beta = nn.Parameter(np.pi*torch.ones(shape))
        self.register_buffer('moving_mag', 1.*torch.ones(shape))
        self.register_buffer('moving_min', np.pi*torch.ones(shape))

    def forward(self, X):
        if self.moving_mag.device != X.device:
            self.moving_mag = self.moving_mag.to(X.device)
            self.moving_min = self.moving_min.to(X.device)

        # self.moving_mag = self.moving_mag.to(X.device)
        # self.moving_min = self.moving_min.to(X.device)

        if self.alpha.device != X.device:
            self.alpha = self.alpha.to(X.device)
        if self.gamma.device != X.device:
            self.gamma = self.gamma.to(X.device)
        if self.beta.device != X.device:
            self.beta = self.beta.to(X.device)

        Y, self.moving_mag, self.moving_min = stretch(
            X,
            self.alpha,
            self.gamma,
            self.beta,
            self.moving_mag,
            self.moving_min,
            eps=1e-5,
            momentum=0.99,
            training = self.training)
        
        return Y
    
class DAE_Layer(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_dims: int,
                 alpha: list):
        super().__init__()

        self.latent_dims = num_features
        self.stretch = Stretch(num_features=num_features, num_dims=num_dims, alpha=alpha)

    def forward(self, z):

        s = self.stretch(z)
        s = gaussian_interpolation(s)
        s = euler_encoding(s, self.latent_dims)

        return s