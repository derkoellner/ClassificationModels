import torch
from torch.utils.data import DataLoader

import wandb

from Models.Activation_AE.ReadyToGoAEs import StackedAE

def train_SAE(model: StackedAE,
          n_layers: int,
          epochs: int,
          training_data: DataLoader,
          test_data: DataLoader,
          optimizer,
          signal_loss_fn,
          feat_loss_fn,
          device,
          log: bool = True):
    
    size = len(training_data.dataset)
    batch_size = training_data.batch_size
    num_batches = len(test_data)
    
    for layer in range(n_layers):

        print(f'-------------------------\nLayer [{layer+1}/{n_layers}]')

        for epoch in range(epochs):

            model.requires_grad_(False)
            model.Encoder.Encoder_Layers[layer].requires_grad_(True)
            model.Decoder.Decoder_Layers[-(layer+1)].requires_grad_(True)

            print(f"###########################\nEpoch {epoch + 1}/{epochs}\n---------------------------")

            running_loss, running_feature_loss, running_signal_loss = 0.0, 0.0, 0.0

            for batch_idx, batch in enumerate(training_data):
                signal = batch
                signal = signal.to(device)

                optimizer.zero_grad()

                if layer != 0:
                    signal_hat, features, features_hat = model(signal, layer)
                    signal_loss = signal_loss_fn(signal_hat, signal)
                    feature_loss = feat_loss_fn(features_hat, features)

                    loss = signal_loss + feature_loss

                    running_signal_loss += signal_loss.item()
                    running_feature_loss += feature_loss.item()

                elif layer == 0:
                    signal_hat = model(signal)
                    loss = signal_loss_fn(signal_hat, signal)
                    running_signal_loss += loss.item()

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                current = (batch_idx + 1) * batch_size
                current = current if current <= size else size

                print(f'loss: {loss.item():>7f} [{current:>5d}/{size:>5d}]')

            avg_train_loss = running_loss / num_batches
            running_signal_loss /= num_batches
            running_feature_loss /= num_batches

            test_loss, test_signal_loss, test_feature_loss = test_SAE(model=model, test_data=test_data, loss_fn=signal_loss_fn, device=device, layer=layer)

            print(f"Train Loss: {avg_train_loss:.4f}\nTest Loss: {test_loss:>8f}")

            if log:
                wandb.log(
                    {'train-loss': avg_train_loss,
                     'train-signal-loss': running_signal_loss,
                     'train-feature-loss': running_feature_loss,
                    'test-loss': test_loss,
                    'test-signal-loss': test_signal_loss,
                    'test-feature-loss': test_feature_loss}
                )

def test_SAE(model: StackedAE,
             test_data: DataLoader,
             loss_fn,
             device,
             layer: int):
    
    num_batches = len(test_data)
    # model.eval()
    model.requires_grad_(False)
    test_loss, running_feature_loss, running_signal_loss = 0.0, 0.0, 0.0

    with torch.no_grad():

        for signal in test_data:
            signal = signal.to(device)

            if layer != 0:
                signal_hat, features, features_hat = model(signal, layer)
                signal_loss = loss_fn(signal_hat, signal).item()
                feature_loss = loss_fn(features_hat, features).item()

                test_loss += signal_loss + feature_loss

                running_signal_loss += signal_loss
                running_feature_loss += feature_loss

            elif layer == 0:
                signal_hat = model(signal)
                loss = loss_fn(signal_hat, signal)
                test_loss += loss.item()
                running_signal_loss += loss.item()

    test_loss /= num_batches
    running_signal_loss /= num_batches
    running_feature_loss /= num_batches

    return test_loss, running_signal_loss, running_feature_loss

