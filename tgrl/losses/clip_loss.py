import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class CLIPLoss(nn.Module):
    def __init__(self, temperature: float = 0.2, lamda: float = 1.0):
        """
        Initialize the CLIP loss module.

        Args:
        - temperature (float): A scaling factor used in the softmax function.
        """
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, text_features, image_features):
        """
        Forward pass for computing the CLIP loss.

        Args:
        - text_features (torch.Tensor): A tensor of shape (batch_size, feature_size) representing the text embeddings.
        - image_features (torch.Tensor): A tensor of shape (batch_size, feature_size) representing the image embeddings.

        Returns:
        - torch.Tensor: The scalar loss for the given batch of text and image features.
        """
        # Normalize the features to unit length
        text_features = F.normalize(text_features, p=2, dim=1)
        image_features = F.normalize(image_features, p=2, dim=1)

        # Compute the similarity matrix
        similarity = torch.matmul(text_features, image_features.T) / self.temperature

        # Labels for each entry in the batch
        labels = torch.arange(similarity.size(0), device=similarity.device)

        # Compute the loss for both directions
        # Text-to-Image loss
        loss_ti = F.cross_entropy(similarity, labels)
        # Image-to-Text loss
        loss_it = F.cross_entropy(similarity.T, labels)

        # The total loss is the average of the two losses
        total_loss = (loss_ti + loss_it) / 2

        return total_loss
