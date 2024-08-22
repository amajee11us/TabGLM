import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from pytorch_metric_learning.losses import SelfSupervisedLoss, SupConLoss


class SimCLR(nn.Module):
    def __init__(self, temperature: float = 0.2, lamda: float = 1.0):
        """Simple Implementation of the Contrastive Loss from SCARF: https://arxiv.org/pdf/2106.15147

        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: Tensor, z_j: Tensor):
        """
        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """
        batch_size = z_i.size(0)

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity, batch_size)
        sim_ji = torch.diag(similarity, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (
            ~torch.eye(
                batch_size * 2, batch_size * 2, dtype=torch.bool, device=z_i.device
            )
        ).float()
        numerator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity / self.temperature)

        all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)

        return loss
