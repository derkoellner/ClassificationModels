import numpy as np

import torch

from sklearn.decomposition import PCA

"""
1.  Stretch (Normalization)
2.  Gaussian Interpolation
3.  Euler Encoding
"""

def compute_lambda(X: np.ndarray, alpha: float, n_subgroups: int) -> np.ndarray:
    if n_subgroups <= 2:
        return np.array([1.0, alpha])
    
    num_trials, _, _ = X.shape
    X_flat = X.reshape(num_trials, -1)

    pca = PCA(n_components=n_subgroups)
    pca.fit(X_flat)

    S = pca.singular_values_

    if len(S) < n_subgroups:
        S = np.concatenate([S, alpha * np.ones(n_subgroups - len(S))])

    # Normalize by max
    S_norm = S / np.max(S)
    # Round to one decimal
    S_round = np.round(S_norm, 1)
    # Replace values < 1 with alpha
    S_round[S_round < 1] = alpha
    return S_round


# Stretch/Normalization Layer
def stretch(X, alpha, gamma, beta, moving_mag, moving_min, eps, momentum, training):
    '''
    the code is based on the batch normalization in
    http://preview.d2l.ai/d2l-en/master/chapter_convolutional-modern/batch-norm.html
    '''
    if not training:
        X_hat = (X - moving_min)/moving_mag
    else:
        # assert len(X.shape) in (2, 4)
        min_ = X.min(dim=0)[0]
        max_ = X.max(dim=0)[0]

        mag_ = max_ - min_
        X_hat =  (X - min_)/mag_
        moving_mag = momentum * moving_mag + (1.0 - momentum) * mag_
        moving_min = momentum * moving_min + (1.0 - momentum) * min_
    Y = (X_hat*gamma*alpha) + beta
    return Y, moving_mag.data, moving_min.data
    
# Gaussian Interpolation
def gaussian_interpolation(z):
    diff = torch.abs(z - z.unsqueeze(axis = 1))
    none_zeros = torch.where(diff == 0., torch.tensor([100.]).to(z.device), diff)
    z_scores,_ = torch.min(none_zeros, axis = 1)
    std =  torch.normal(mean = 0., std = 1.*z_scores).to(z.device)
    s = z + std
    return s

# Euler Encoding
def euler_encoding(s, latent_dim):
    c = torch.cat((torch.cos(2*np.pi*s), torch.sin(2*np.pi*s)), -1)
    # c = c.T.reshape(latent_dim*2,-1).T

    return c