import torch
import torch.nn as nn
import wandb

import os
# from pathlib import Path

def log_model(model: nn.Module,
              model_path: str,
              model_name: str):
    """
    Logs a PyTorch model to Weights & Biases.

    Args:
    model: Module - The model to log.
    model_path: str - Path to save the model state dictionary.
    model_name: str - Name for the logged model artifact in wandb.

    Returns:
    None - Logs the model artifact to wandb.
    """
    # WANDB_DIR = Path(__file__).resolve().parents[2]
    # model_path = os.path.join(WANDB_DIR, model_path, f'{model_name}.pth')
    model_path = os.path.join(model_path, f'{model_name}.pth')

    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

    artifact = wandb.Artifact(model_name, type='model')
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)