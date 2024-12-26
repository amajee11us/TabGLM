import torch
import torch.nn as nn
import numpy as np
import wandb
from tabglm.utils import log_metrics

def train_one_epoch(
    model,
    trainloader,
    supervised_criterion,
    consistency_criterion,
    optimizer,
    task_type,
    epoch,
    device,
    consistency=True,
    is_supervised=True,
    alpha=0.5,
):
    model.train()
    running_loss = 0.0
    phase = "train"
    all_labels, all_predictions, all_probabilities = [], [], []

    for graph_in, text_in, labels in trainloader:
        graph_in, labels = graph_in.to(device), labels.to(device)
        text_in = {key: value.to(device) for key, value in text_in.items()}

        optimizer.zero_grad()

        if is_supervised:
            graph_features, text_features, logits = model(text_in, graph_in, labels)
            # Supervised Loss is calculated on only one augmentation of the input image
            loss, predicted, probabilities = calculate_loss_and_predictions(
                logits, labels, task_type, supervised_criterion
            )
        else:
            graph_features, text_features = model(text_in, graph_in, labels)
            loss = 0.0

        if consistency:
            consistency_loss = calculate_consistency_loss(
                graph_features, text_features, consistency_criterion
            )

            loss = (1 - alpha) * loss + alpha * consistency_loss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().detach().numpy())
        if probabilities is not None:
            all_probabilities.extend(probabilities.cpu().detach().numpy())

    average_loss = running_loss / len(trainloader)
    log_metrics(
        task_type,
        all_labels,
        all_predictions,
        all_probabilities,
        average_loss,
        "train",
        epoch,
    )


def calculate_loss_and_predictions(outputs, labels, task_type, criterion):
    outputs = outputs.squeeze(1)
    if task_type == "multi_class":
        probabilities = torch.softmax(outputs, dim=1)
        predicted = torch.max(probabilities, 1)[1]
        loss = criterion(outputs, labels.long())
    elif task_type == "binary":
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities > 0.5).float()
        loss = criterion(outputs, labels.float())
    elif task_type == "regression":
        predicted = outputs
        loss = criterion(predicted, labels.float())
        probabilities = None
    else:
        raise ValueError("Unknown task type!")

    return loss, predicted, probabilities


def calculate_consistency_loss(op_modality_1, op_modality_2, loss_function=None):
    # Compute consistency loss for each pair
    consistency_loss = loss_function(op_modality_1, op_modality_2)

    # Compute average consistency loss
    # consistency_loss = sum(consistency_loss) / len(consistency_loss)
    return consistency_loss


def evaluate_model(
    model,
    dataloader,
    criterion,
    task_type,
    device,
    phase,
    epoch=None,
    scheduler=None,
    is_supervised=False,
):
    model.eval()
    running_loss = 0.0
    all_predictions, all_labels, all_probabilities = [], [], []

    with torch.no_grad():
        for graph_in, text_in, labels in dataloader:
            graph_in, labels = graph_in.to(device), labels.to(device)
            text_in = {key: value.to(device) for key, value in text_in.items()}

            _, _, outputs = model(text_in, graph_in)

            loss, predicted, probabilities = calculate_loss_and_predictions(
                outputs, labels, task_type, criterion
            )

            running_loss += loss.item()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().detach().numpy())
            if probabilities is not None:
                all_probabilities.extend(probabilities.cpu().detach().numpy())

    average_loss = running_loss / len(dataloader)
    metrics = log_metrics(
        task_type,
        all_labels,
        all_predictions,
        all_probabilities,
        average_loss,
        phase,
        epoch,
    )

    if phase not in ["train", "val", "test"]:
        raise ValueError("phase should be either train, val, or test")

    if phase == "val" and scheduler is not None:
        scheduler.step(average_loss)

    return average_loss, metrics
