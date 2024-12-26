import torch
import torch.nn as nn
import numpy as np

from tabglm.losses.clip_loss import CLIPLoss
from tabglm.losses.facility_location import FL
from tabglm.losses.graph_cut import GC
from tabglm.losses.simclr import SimCLR

# TODO : Merge into existing class of GC
from pytorch_metric_learning.losses import SelfSupervisedLoss, SupConLoss

"""
When adding new objectives, make sure you add their names to this list.
"""
loss_dict = {
    "simclr": SimCLR,
    "fl": FL,
    "gc": SelfSupervisedLoss,
    "clip": CLIPLoss,
    "mse": nn.MSELoss,
}


def get_consistency_criterion(function_type="fl"):
    """
    Factory function to chose the combinatorial loss
    type : Can be fl (Facility-Location), gc (Graph-Cut) or logdet (Log-Determinant)
    """
    if function_type not in list(loss_dict.keys()):
        raise Exception("Objective function of type : {}, does not exist.".format(type))

    if function_type == "mse":
        return loss_dict[function_type]()

    return loss_dict[function_type](temperature=0.1, lamda=1.0)


def get_criterion(task_type, y_train_enc):
    if task_type == "binary":
        positive_weight = np.sum(y_train_enc == 0) / np.sum(y_train_enc == 1)
        class_weights = torch.tensor([positive_weight], dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    elif task_type == "multi_class":
        num_classes = len(np.unique(y_train_enc))
        class_counts = np.bincount(y_train_enc)
        class_weights = 1.0 / (class_counts.astype(np.float32) + 1e-6)
        class_weights_tensor = torch.tensor(class_weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    elif task_type == "regression":
        criterion = nn.MSELoss()

    return criterion
