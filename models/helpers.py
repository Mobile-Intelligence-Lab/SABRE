import os
import torch

from utils.log import printer


def load_model(model, model_path, log_path=None):
    """
    Load model weights from a specified path.

    Parameters:
    - model: The model to be updated.
    - model_path: Path to the pretrained model file.

    Returns:
    - The model, potentially loaded with pretrained weights.
    """
    if os.path.exists(model_path):
        load_pretrained_weights(model, model_path, log_path)
    return model


def load_pretrained_weights(model, model_path, log_path=None):
    """
    Load pretrained weights into the model if the model file exists.

    Parameters:
    - model: The model instance into which weights will be loaded.
    - model_path: Path to the pretrained model file.
    """
    if not os.path.exists(model_path):
        printer(f"Model {model_path} not found.", log_path)
        return

    printer(f"Loading model weights from: {model_path}", log_path)
    states = torch.load(model_path, map_location=lambda storage, loc: storage)
    updated_state_dict = remove_module_prefix(states)
    model.load_state_dict(updated_state_dict, strict=False)


def remove_module_prefix(states):
    """
    Remove the 'module.' prefix from state dict keys, if present.

    Parameters:
    - states: The state dictionary from which to remove the prefix.

    Returns:
    - A new state dictionary with the 'module.' prefix removed from keys.
    """
    return {key.replace("module.", ""): value for key, value in states.items()}


def save_model(model, save_path, log_path=None):
    """
    Save a trained model's state dictionary on the CPU at the provided path.

    Parameters:
    - model: The trained model to be saved.
    - save_path: The path where the model's state dictionary will be saved.
    """
    # Ensure the model is in evaluation mode.
    model.eval()

    # Move the model to CPU (in case it was on a GPU).
    model_to_save = model.to('cpu')

    # Save the model's state dictionary.
    torch.save(model_to_save.state_dict(), save_path)

    printer(f"Model saved successfully: {save_path}", log_path)
