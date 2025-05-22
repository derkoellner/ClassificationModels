import torch.nn as nn
from torch.utils.data import DataLoader

from ..wandb_functions import log_model
from .basic_training_functions import train_test
from .stacked_training_functions import train_SAE

from Models.Activation_AE.ReadyToGoAEs import StackedAE

import wandb

def train_model(
                model: nn.Module,
                train_dataloader: DataLoader,
                test_dataloader: DataLoader,
                optimizer,
                loss_fn,
                device,
                epochs: int,
                model_name: str,
                wandb_config: dict,
                direct_decode: bool = True,
                project='Activation_AE'
                ):

    wandb.init(
        project=project,
        name=model_name,

        config=wandb_config
    )

    train_test(
        model=model,
        epochs=epochs,
        train_data=train_dataloader,
        test_data=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        direct_decode=direct_decode
    )

    log_model(
        model=model,
        model_path='wandb/models/',
        model_name=model_name)
    
    wandb.finish()

def train_test_SAE(model: StackedAE,
          n_layers: int,
          epochs: int,
          training_data: DataLoader,
          test_data: DataLoader,
          model_name: str,
          optimizer,
          signal_loss_fn,
          feat_loss_fn,
          device,
          wandb_config,
          project='Activation_AE'):
    
    wandb.init(
        project=project,
        name='StackedAE',

        config=wandb_config
    )

    train_SAE(
        model=model,
        n_layers=n_layers,
        epochs=epochs,
        training_data=training_data,
        test_data=test_data,
        optimizer=optimizer,
        signal_loss_fn=signal_loss_fn,
        feat_loss_fn=feat_loss_fn,
        device=device,
        log=True
    )

    log_model(
        model=model,
        model_path='wandb/models/',
        model_name=model_name)
    
    wandb.finish()