import math

import torch
import torch.nn as nn
import torch.optim as optim

from core.attacks import PGD, EoTPGD
from models.helpers import save_model
from utils.log import printer

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ceil_dec(number, decimals: int = 3):
    factor = 10 ** decimals
    return math.ceil(number * factor) / factor


def floor_dec(number, decimals: int = 3):
    factor = 10 ** decimals
    return math.floor(number * factor) / factor


def adjust_learning_rate(optimizer, epoch, n_epochs, initial_lr):
    """
    Adjusts the learning rate during training.

    Parameters:
    - optimizer: The optimizer for which to adjust the learning rate.
    - epoch: The current epoch number.
    - n_epochs: The total number of epochs.
    - initial_lr: The initial learning rate.
    """
    lr = initial_lr * (0.1 ** (epoch // (0.5 * n_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_normalizers(data_loader, n_channels=1):
    means = torch.zeros(n_channels, device=device)
    squared_means = torch.zeros(n_channels, device=device)
    batch_count = len(data_loader)

    with torch.no_grad():
        for (inputs, targets) in data_loader:
            x = inputs.to(device)
            means += x.mean(dim=(0, 2, 3))
            squared_means += (x ** 2).mean(dim=(0, 2, 3))
        means /= batch_count
        stds = (squared_means / batch_count - means ** 2) ** .5
        mean = means.reshape(-1, 1, 1).to(device)
        stds = stds.reshape(-1, 1, 1).to(device)

    return mean, stds


def set_normalizers(model, train_data_loader, n_channels):
    mean, std = compute_normalizers(train_data_loader, n_channels)
    if hasattr(model, 'set_normalizers'):
        model.set_normalizers((mean, std))
    return mean, std


def baseline_train(model, data_loader, n_epochs=100, learning_rate=1e-3, save_path=None, log_path=None, params={}):
    """
    Trains a PyTorch model.

    Parameters:
    - model: The PyTorch model to be trained.
    - data_loader: DataLoader for the training data.
    - n_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - save_path: Path to save the model after training. If None, the model is not saved.
    - log_path: Path to a log file where progress and results will be saved. If None, only prints to stdout.
    - params: Additional parameters for adversarial training.
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    normalize, n_channels = params.get("normalize", False), params.get("n_channels", 1)
    mean, std = compute_normalizers(data_loader, n_channels) if normalize else (None, None)

    for epoch in range(n_epochs):
        model.train()
        model.to(device)
        running_loss = 0.0
        correct, total = 0, 0
        batch_count = 0

        adjust_learning_rate(optimizer, epoch, n_epochs, learning_rate)

        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = inputs.float().to(device), labels.to(device)

            if normalize:
                inputs = (inputs - mean) / std

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct, total = update_accuracy(outputs, labels, correct, total)

            batch_count += 1

            if i % 10 == 9:  # Print every 10 mini-batches
                printer(f"[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {running_loss / batch_count:.3f} | "
                        f"Accuracy: {correct / total:.3f} ({correct}/{total})", log_path)

        printer(f"Epoch {epoch + 1} completed, Average Loss: {running_loss / len(data_loader):.3f} | "
                f"Accuracy: {correct / total:.3f} ({correct}/{total})", log_path)

        # Save the model
        if save_path is not None:
            save_model(model, save_path, log_path)

    printer('Finished Training', log_path)

    # Save the trained model
    if save_path is not None:
        save_model(model, save_path)

    return model


def adversarial_train(model, data_loader, n_epochs=100, learning_rate=1e-3, save_path=None, log_path=None, params={}):
    """
    Performs adversarial training.

    Parameters:
    - model: The PyTorch model to be trained.
    - data_loader: DataLoader for the training data.
    - n_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - save_path: Path to save the model after training. If None, the model is not saved.
    - log_path: Path to a log file where progress and results will be saved. If None, only prints to stdout.
    - params: Additional parameters for adversarial training.
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    normalize, n_channels = params.get("normalize", False), params.get("n_channels", 1)
    pgd_eps, pgd_alpha, pgd_steps = params.get("eps", 8. / 255), params.get("alpha", 2. / 255), params.get("steps", 7)
    attack = PGD(eps=pgd_eps, alpha=pgd_alpha, steps=pgd_steps, bounds=(0, 1))
    mean, std = compute_normalizers(data_loader, n_channels) if normalize else (None, None)

    for epoch in range(n_epochs):
        model.train()
        model.to(device)
        running_loss = 0.0
        correct, total = 0, 0
        batch_count = 0

        adjust_learning_rate(optimizer, epoch, n_epochs, learning_rate)

        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = inputs.float().to(device), labels.to(device)

            if normalize:
                inputs = (inputs - mean) / std

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            attack.set_model(model)
            adversarial_inputs = attack(inputs, labels.clone()).float().to(device)

            optimizer.zero_grad()
            outputs_adv = model(adversarial_inputs)
            loss_adv = criterion(outputs_adv, labels)
            loss_adv.backward()
            optimizer.step()

            running_loss += (loss.item() + loss_adv.item()) / 2

            outputs = torch.cat((outputs, outputs_adv), dim=0)
            targets = torch.cat((labels, labels), dim=0)

            correct, total = update_accuracy(outputs, targets, correct, total)

            batch_count += 1
            if i % 10 == 9:  # Print every 10 mini-batches
                printer(f"[Epoch: {epoch + 1}, Batch: {i + 1}] Loss: {running_loss / batch_count:.3f} | "
                        f"Accuracy: {correct / total:.3f} ({correct}/{total})", log_path)

        printer(f"Epoch {epoch + 1} completed, Average Loss: {running_loss / len(data_loader):.3f} | "
                f"Accuracy: {correct / total:.3f} ({correct}/{total})", log_path)

        # Save the model
        if save_path is not None:
            save_model(model, save_path, log_path)

    printer('Finished Training', log_path)

    # Save the trained model
    if save_path is not None:
        save_model(model, save_path, log_path)

    return model


def update_running_losses(running_losses, reconstruction_loss, classification_loss):
    """Updates running losses with the latest batch losses."""
    running_losses['reconstruction'] += reconstruction_loss
    running_losses['classification'] += classification_loss


def update_accuracy(outputs, labels, correct, total):
    """Updates the accuracy metrics based on the current batch."""
    _, predicted = outputs.max(1)
    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
    return correct, total


def print_training_progress(epoch, n_epochs, i, data_loader, batch_count, running_losses, model, total, correct,
                            log_path=None):
    """Prints training progress."""
    lambda_r = model.lambda_r.item() if model.lambda_r.numel() == 1 else 0
    printer(f"[Epoch: {epoch + 1}/{n_epochs}, Batch: {i + 1}/{len(data_loader)}] "
            f"Reconstruction/Classification Loss: {running_losses['reconstruction'] / batch_count:.3f}/"
            f"{running_losses['classification'] / batch_count:.3f} | Lambda_r: {lambda_r:.3f} | "
            f"Accuracy: {100 * correct / total:.2f}% ({correct}/{total})", log_path)


def summarize_epoch(epoch, n_epochs, batch_count, running_losses, model, total, correct, log_path=None):
    """Prints a summary of the epoch's performance."""
    lambda_r = model.lambda_r.item() if model.lambda_r.numel() == 1 else 0
    printer(f"[Epoch {epoch + 1}/{n_epochs} completed] "
            f"Reconstruction/Classification Loss: {running_losses['reconstruction'] / batch_count:.3f}/"
            f"{running_losses['classification'] / batch_count:.3f} | Lambda_r: {lambda_r:.3f} | "
            f"Accuracy: {100 * correct / total:.2f}% ({correct}/{total})", log_path)


def sabre_train(model, data_loader, n_epochs=100, learning_rate=1e-3, save_path=None, log_path=None, params={}):
    """
    Performs Sabre's robust training.

    Parameters:
    - model: The PyTorch model to be trained.
    - data_loader: DataLoader for the training data.
    - n_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    - save_path: Path to save the model after training. If None, the model is not saved.
    - log_path: Path to a log file where progress and results will be saved. If None, only prints to stdout.
    - params: Additional parameters for robust training.
    """
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    normalize, n_channels = params.get("normalize", False), params.get("n_channels", 1)
    plot_images = params.get("plot_images", False)
    eps = params.get("eps", 8. / 255) * 1.25
    attack = EoTPGD(eps=eps, alpha=eps, steps=1, eot_steps=1, bounds=(0, 1))

    if normalize:
        set_normalizers(model, data_loader, n_channels)

    print_every = max(1, len(data_loader) // 10)

    for epoch in range(n_epochs):
        model.train()
        model.to(device)
        running_losses = {'reconstruction': 0.0, 'classification': 0.0}
        correct, total = 0, 0
        batch_count = 0

        adjust_learning_rate(optimizer, epoch, n_epochs, learning_rate)

        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = inputs.float().to(device), labels.to(device)
            original_inputs = inputs.clone()
            batch_size = len(inputs)

            attack.set_model(model)
            adversarial_inputs = attack(inputs, labels.clone()).float()
            inputs = torch.cat((inputs, adversarial_inputs.to(device)), dim=0)

            optimizer.zero_grad()
            reconstructed = model.preprocessing(inputs)
            reconstruction_targets = torch.cat((original_inputs, original_inputs), dim=0).clone().detach()
            reconstruction_loss = 2. * torch.sqrt(reconstruction_criterion(reconstructed, reconstruction_targets))
            reconstruction_loss.backward(retain_graph=True)

            features, outputs = model.features_logits(reconstructed)
            targets = torch.cat((labels, labels), dim=0).clone()
            classifier_loss = criterion(outputs, targets)

            classifier_loss += torch.sqrt(reconstruction_criterion(features[-1][:batch_size],
                                                                   features[-1][batch_size:]))

            classifier_loss.backward()
            optimizer.step()

            # print statistics
            update_running_losses(running_losses, reconstruction_loss.item(), classifier_loss.item())
            correct, total = update_accuracy(outputs, targets, correct, total)

            batch_count += 1

            if i % print_every == print_every - 1:
                print_training_progress(epoch, n_epochs, i, data_loader, batch_count, running_losses, model, total,
                                        correct, log_path)

                if plot_images:
                    fig, axs = plt.subplots(1, 4, figsize=(8, 4))
                    axs[0].imshow(original_inputs[0].cpu().detach().numpy().transpose((1, 2, 0)))
                    axs[0].axis('off')
                    axs[0].set_title('Benign')

                    axs[1].imshow(adversarial_inputs[0].cpu().detach().numpy().transpose((1, 2, 0)))
                    axs[1].axis('off')
                    axs[1].set_title('Adversarial')

                    axs[2].imshow(reconstructed[0].cpu().detach().numpy().transpose((1, 2, 0)))
                    axs[2].axis('off')
                    axs[2].set_title('Reconstructed\nBenign')

                    axs[3].imshow(reconstructed[batch_size].cpu().detach().numpy().transpose((1, 2, 0)))
                    axs[3].axis('off')
                    axs[3].set_title('Reconstructed\nAdversarial')
                    plt.show()

        summarize_epoch(epoch, n_epochs, batch_count, running_losses, model, total, correct, log_path)

        # Save the model
        if save_path is not None:
            save_model(model, save_path, log_path)

    printer('Finished Training', log_path)

    # Save the trained model
    if save_path is not None:
        save_model(model, save_path, log_path)

    return model


def evaluate(model, attack, data_loader, log_path=None):
    """
    Evaluate the model's performance on a dataset, optionally using an adversarial attack.

    Parameters:
    - model (torch.nn.Module): The model to evaluate.
    - attack (callable): An optional attack function that takes inputs and labels and returns adversarial examples.
    - data_loader (torch.utils.data.DataLoader): The DataLoader providing the dataset for evaluation.
    - log_path (str, optional): Path to a log file where progress and results will be saved. If None, only prints to stdout.
    - params: Additional training parameters.

    Returns:
    - Tuple[int, int]: The number of correct predictions and the total number of samples evaluated.
    """
    model.to(device)
    model.eval()

    if attack is not None and hasattr(attack, 'set_model'):
        attack.set_model(model)
    else:
        attack = lambda x, y: x

    correct, total = 0, 0
    print_every = max(1, len(data_loader) // 10)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader, 0):
            inputs, labels = inputs.float().to(device), labels.to(device)
            shape = inputs.shape

            with torch.enable_grad():
                inputs = attack(inputs, labels).float()

            if isinstance(inputs, list):
                inputs = torch.stack(inputs).reshape(*shape)

            outputs = model(inputs)
            correct, total = update_accuracy(outputs, labels, correct, total)

            if i % print_every == print_every - 1:
                printer(f"[Batch: {i + 1} / {len(data_loader)}] "
                        f"Accuracy: {100 * correct / total:.3f} ({correct}/{total})", log_path)

    printer('Finished Evaluation', log_path)
    printer(f"Accuracy: {100 * correct / total:.3f} ({correct}/{total})", log_path)

    return correct, total
