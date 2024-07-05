import os
import csv
import pandas as pd
import torch
import random
import yaml
from numpy.random import RandomState, SeedSequence, MT19937
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, r2_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import wandb

from transformers.utils import logging

logging.set_verbosity_error()

"""
Set the seed values for consistent performance metrics
"""


def set_seed(seed, disable_cudnn=False):
    torch.manual_seed(seed)  # Seed the RNG for all devices (both CPU and CUDA).
    torch.cuda.manual_seed_all(
        seed
    )  # If you are using multi-GPU. In case of one GPU, you can use # torch.cuda.manual_seed(SEED).
    np.random.seed(seed)
    random.seed(seed)  # Set python seed for custom operators.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    rs = RandomState(
        MT19937(SeedSequence(seed))
    )  # If any of the libraries or code rely on NumPy seed the global NumPy RNG.

    if not disable_cudnn:
        torch.backends.cudnn.benchmark = (
            False  # Causes cuDNN to deterministically select an algorithm,
        )
        # possibly at the cost of reduced performance
        # (the algorithm itself may be nondeterministic).
        torch.backends.cudnn.deterministic = (
            True  # Causes cuDNN to use a deterministic convolution algorithm,
        )
        # but may slow down performance.
        # It will not guarantee that your training process is deterministic
        # if you are using other libraries that may use nondeterministic algorithms
    else:
        torch.backends.cudnn.enabled = (
            False  # Controls whether cuDNN is enabled or not.
        )
        # If you want to enable cuDNN, set it to True.


# def normalize_adj_matrix(adj):
#     D = torch.sum(adj, 0)
#     D_hat = torch.diag(((D) ** (-0.5)))
#     adj_normalized = torch.mm(torch.mm(D_hat, adj), D_hat)
#     return adj_normalized


def find_repo_root(path="."):
    current_path = Path(path).resolve()
    # Check the current path itself before checking parents
    if (current_path / ".git").exists():
        return current_path
    for parent in current_path.parents:
        if (parent / ".git").exists():
            return parent
    return None


def get_config(config_path):
    with open(config_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    data_config = config["data_config"]
    fit_config = config["fit_config"]
    return data_config, fit_config


def to_dense_array(series):
    # Check if the series is stored as a sparse array
    if isinstance(series.array, pd.arrays.SparseArray):
        # Convert SparseArray to a dense NumPy array
        return series.sparse.to_dense().to_numpy()
    else:
        # If it's not a sparse array, just convert to NumPy array directly
        return series.to_numpy()


# def create_dataloaders(dataset, phase, batch_size):
#     # Create dataloaders
#     shuffle = phase == "train"
#     setloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=False,
#         num_workers=12,
#         pin_memory=True,
#     )
#     return setloader


# def get_criterion(task_type, y_train_enc):
#     if task_type == "binary":
#         positive_weight = np.sum(y_train_enc == 0) / np.sum(y_train_enc == 1)
#         class_weights = torch.tensor([positive_weight], dtype=torch.float)
#         criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
#     elif task_type == "multi_class":
#         num_classes = len(np.unique(y_train_enc))
#         class_counts = np.bincount(y_train_enc)
#         class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
#         class_weights_tensor = torch.tensor(class_weights)
#         criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
#     elif task_type == "regression":
#         criterion = nn.MSELoss()

#     return criterion


# def get_consistency_criterion(consistency_loss_type):
#     # Validate if the loss named in consistency_loss_type is supported
#     if consistency_loss_type in supported_combinatorial_losses:
#         loss_function_name = consistency_loss_type
#         loss_type = "comb"  # Single keyword indicating the combinatorial losses
#     elif consistency_loss_type == "mse":
#         loss_type = "mse"
#     else:
#         raise Exception(
#             "Loss Type: {}, does not match any known types: Choose between [mse, comb]".format(
#                 consistency_loss_type
#             )
#         )

#     # Chose the loss function based on the function type
#     if loss_type == "mse":
#         # MSE Loss instance for consistency loss
#         consistency_criterion = nn.MSELoss()
#         return consistency_criterion
#     elif loss_type == "comb":
#         consistency_criterion = combinatorial_loss(function_type=loss_function_name)
#         return consistency_criterion


def store_metrics_as_csv(metrics, save_path="output_metrics.csv"):
    """
    metrics   :  A JSON formatted dictionary with key value pairs.
    save_path :  A path (str) determining the path where metrics
                 are stored.
    """
    # Check if file exists to determine write mode
    file_exists = os.path.isfile(save_path)

    # Open the file in append mode if it exists, otherwise write mode
    with open(save_path, "a" if file_exists else "w", newline="") as csvfile:
        # Get the keys from the metrics dictionary to use as headers
        fieldnames = metrics.keys()

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Write the metrics as a row
        writer.writerow(metrics)

    print(f"Stored updated metrics at {save_path}.")


def log_metrics(task_type, labels, predictions, probabilities, loss, phase, epoch=None):
    """
    Calculate and log metrics for the given task type, distinguishing between training and validation phases.

    Parameters:
        task_type (str): The type of task ('binary', 'multi_class', or 'regression').
        labels (list): The true labels.
        predictions (list): The predicted labels or values.
        probabilities (list): The probabilities associated with predictions (for classification).
        loss (float): The average loss over the epoch.
        epoch (int): The current epoch number.
        phase (str): The phase of training, either 'train' or 'val'.

    Returns:
        dict: A dictionary containing logged metrics.
    """
    prefix = f"{phase}_"  # Prefix to distinguish between train and val metrics
    metrics = {}
    if epoch:
        metrics = {f"{prefix}loss": loss, "epoch": epoch}

    if task_type == "regression":
        r2 = r2_score(labels, predictions) * 100
        metrics[f"{prefix}r2"] = r2
        print(f"Epoch {epoch}: {phase.capitalize()} Loss {loss:.4f}, R^2 {r2:.2f}%")

    elif task_type in ["binary", "multi_class"]:
        accuracy = accuracy_score(labels, predictions) * 100
        f1 = f1_score(labels, predictions, average="macro") * 100
        metrics[f"{prefix}accuracy"] = accuracy
        metrics[f"{prefix}f1"] = f1
        # metrics.update({f'{prefix}accuracy': accuracy, f'{prefix}f1': f1})

        if probabilities:
            if task_type == "binary":
                auroc = roc_auc_score(labels, probabilities) * 100
            else:
                classes = np.unique(labels)
                y_bin = label_binarize(labels, classes=classes)
                auroc = (
                    roc_auc_score(
                        y_bin,
                        np.array(probabilities),
                        multi_class="ovr",
                        average="macro",
                    )
                    * 100
                )
            metrics[f"{prefix}auroc"] = auroc
            print(
                f"Epoch {epoch}: {phase.capitalize()} Loss {loss:.4f}, Accuracy {accuracy:.2f}%, F1 {f1:.2f}%, AUROC {auroc:.2f}%"
            )
        else:
            print(
                f"Epoch {epoch}: {phase.capitalize()} Loss {loss:.4f}, Accuracy {accuracy:.2f}%, F1 {f1:.2f}%"
            )

    # Log metrics to wandb or any other experiment tracking system
    wandb.log(metrics)
    return metrics
