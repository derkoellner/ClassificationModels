import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

def train_test(
        model: nn.Module,
        epochs: int,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer,
        loss_fn,
        device,
        direct_decode: bool = True,
        log: bool = True
    ):
    
    for epoch in range(epochs):

        print(f"###########################\nEpoch {epoch + 1}/{epochs}\n---------------------------")

        model.train(True)
        avg_train_loss = train(model, train_data, optimizer, loss_fn, device, direct_decode)
        model.eval()
        test_loss = test(model, test_data, device, loss_fn, direct_decode)
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
        direct_decode: bool = True
    ):

    size = len(data.dataset)
    batch_size = data.batch_size
    running_loss = 0.0

    for batch_idx, batch in enumerate(data):
        if direct_decode:
            signal = batch
            signal = signal.to(device)

            optimizer.zero_grad()

            signal_hat = model(signal)
            loss = loss_fn(signal_hat, signal)

        else:
            signal, y = batch
            signal, y = signal.to(device), y.to(device)

            signal_hat = model(signal, y) # TODO remove y
            loss = loss_fn(signal_hat, y)

        current = (batch_idx + 1) * batch_size
        current = current if current <= size else size

        print(f'loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]')

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(data)

    return avg_loss

def test(model,
         data: DataLoader,
         device,
         loss_fn, 
         direct_decode: bool = True):
    num_batches = len(data)
    model.eval()
    test_loss = 0.0

    with torch.no_grad():

        if direct_decode:
            for signal in data:
                signal = signal.to(device)
                signal_hat = model(signal)

                test_loss += loss_fn(signal_hat, signal).item()

        else:
            for signal, y in data:
                signal, y = signal.to(device), y.to(device)
                signal_hat = model(signal, y) # TODO remove y

                test_loss += loss_fn(signal_hat, y).item()

    test_loss /= num_batches

    return test_loss
