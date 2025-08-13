import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

class TimeSeriesDataset_TwoFeatures(Dataset):
    """
    PyTorch dataset for time series encoding with an Autoencoder.
    It can be used in two ways:
        1. Encoding the Input Data and Decoding back to Output Data
        2. Encoding both Input and Output Data and Decoding back to Output Data
    
    Args:
    spectrogram: np.array - Array of spectrogram data.
    time_data: np.array - Array of Time Series data.
    """
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data 

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        trial = torch.tensor(self.input_data[index], dtype=torch.float)
        output = torch.tensor(self.output_data[index], dtype=torch.float)
        return trial, output

class TimeSeriesDataset(Dataset):
    """
    PyTorch dataset for time series for Autoencoder.
    Input = Output
    
    Args:
    data: np.array - Array of time series data.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        trial = torch.tensor(self.data[index], dtype=torch.float)
        return trial
    
class DatasetRegression(Dataset):
    """
    PyTorch dataset for time series for Regression.
    
    Args:
    data: np.array - Array of time series data.
    """
    def __init__(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        trial = torch.tensor(self.x[index], dtype=torch.float)
        output = torch.tensor(self.y[index], dtype=torch.float)
        labels = torch.tensor(self.labels[index], dtype=torch.float)
        return trial, output, labels
    
def get_dataloader(input_data: np.ndarray,
                   task: str,
                   output_data: np.ndarray = None,
                   batch_size: int = 32,
                   num_workers: int = 2) -> DataLoader:
    
    if task == "Reconstruction":
    
        if output_data is None:
            dataset = TimeSeriesDataset(input_data)
        elif output_data is not None:
            dataset = TimeSeriesDataset_TwoFeatures(input_data, output_data)

    elif task == "Regression":

        dataset = DatasetRegression(input_data, output_data)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            num_workers=num_workers)
    
    return dataloader