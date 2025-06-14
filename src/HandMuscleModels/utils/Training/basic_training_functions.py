import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

def train_test(
        model: nn.Module,
        epochs: int,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer,
        loss_fn,
        device,
        direct_decode: bool = True,
        log: bool = True,
        accum_steps: int = 1,
        num_input_features: int = 1
    ):
    
    for epoch in range(epochs):

        print(f"###########################\nEpoch {epoch + 1}/{epochs}\n---------------------------")

        model.train(True)
        avg_train_loss = train(
            model,
            train_data,
            optimizer,
            loss_fn,
            device,
            direct_decode,
            accum_steps,
            num_input_features)
        model.eval()
        test_loss = test(
            model,
            test_data,
            device,
            loss_fn,
            direct_decode,
            num_input_features)
        print(f"Train Loss: {avg_train_loss:.4f}\nTest Loss: {test_loss:>8f}")

        if log:
            wandb.log(
                {'train-loss': avg_train_loss,
                 'test-loss': test_loss}
            )

def train(
        model,
        data: DataLoader,
        optimizer,
        loss_fn,
        device,
        direct_decode: bool = True,
        accum_steps: int = 1,
        num_input_features: int = 1
    ):

    size = len(data.dataset)
    batch_size = data.batch_size
    running_loss = 0.0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(tqdm(data, desc="Training", unit="batch")):
        if direct_decode:
            signal = batch
            signal = signal.to(device)

            signal_hat = model(signal)
            loss = loss_fn(signal_hat, signal)

        elif num_input_features == 1:
            signal, y = batch
            signal, y = signal.to(device), y.to(device)

            signal_hat = model(signal)

            loss = loss_fn(signal_hat, y)

        elif num_input_features == 2:
            signal, y = batch
            signal, y = signal.to(device), y.to(device)

            signal_hat = model(signal, y)

            loss = loss_fn(signal_hat, y)

        if hasattr(loss_fn, 'reduction') and loss_fn.reduction != 'mean':
            loss = loss.mean()

        current = (batch_idx + 1) * batch_size
        current = current if current <= size else size

        loss = loss / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(data):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accum_steps

        if (batch_idx + 1) % (10 * accum_steps) == 0:
            current = (batch_idx + 1) * batch_size
            current = current if current <= size else size
            tqdm.write(f'  [Batch {current}/{size}] loss: {running_loss / (batch_idx+1):.6f}')

    avg_loss = running_loss / len(data)

    return avg_loss

def test(model,
         data: DataLoader,
         device,
         loss_fn, 
         direct_decode: bool = True,
         num_input_features: int = 1):
    num_batches = len(data)
    model.eval()
    test_loss = 0.0

    with torch.no_grad():

        if direct_decode:
            for signal in data:
                signal = signal.to(device)
                signal_hat = model(signal)

                loss = loss_fn(signal_hat, signal)
                if hasattr(loss_fn, 'reduction') and loss_fn.reduction != 'mean':
                    loss = loss.mean()
                test_loss += loss.item()

        elif num_input_features == 1:
            for signal, y in data:
                signal, y = signal.to(device), y.to(device)
                signal_hat = model(signal)

                loss = loss_fn(signal_hat, y)
                if hasattr(loss_fn, 'reduction') and loss_fn.reduction != 'mean':
                    loss = loss.mean()
                test_loss += loss.item()

        elif num_input_features == 2:
            for signal, y in data:
                signal, y = signal.to(device), y.to(device)
                signal_hat = model(signal, y)

                loss = loss_fn(signal_hat, y)
                if hasattr(loss_fn, 'reduction') and loss_fn.reduction != 'mean':
                    loss = loss.mean()
                test_loss += loss.item()

    test_loss /= num_batches

    return test_loss
