import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class FL(nn.Module):

    def __init__(self, temperature: float = 0.2, lamda: float = 1.0):
        """Args:
        tempearture: a constant to be divided by consine similarity to enlarge the magnitude
        lamda: Contribution of GC value. Should be > 0.5 to maintain submodularity.
        """
        super().__init__()
        self.temperature = temperature
        self.lamda = lamda

        # scaling parameter - we keep fixed
        self.base_temperature = 0.07

    def _soft_max(self, similarity_matrix, axis=0):
        """
        Approximates the maximum value along a specified axis of a similarity matrix using the log-sum-exp trick.

        Parameters:
        similarity_matrix (Tensor): An n x n similarity matrix.
        axis (int): The axis along which to compute the maximum. Default is 0.

        Returns:
        Tensor: The approximated maximum values along the specified axis.
        """
        # Find the maximum value along the specified axis
        m = torch.max(similarity_matrix, axis, keepdim=True).values

        # Compute the log-sum-exp
        max_approx = m + torch.log(
            torch.sum(torch.exp(similarity_matrix - m), axis, keepdim=True)
        )

        return max_approx.squeeze()

    def forward(self, z_i: Tensor, z_j: Tensor):
        """
        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples

        Returns:
            float: loss
        """

        batch_size = z_i.size(0)
        labels = torch.arange(z_i.shape[0])

        assert z_i.shape[0] == z_j.shape[0]  # Check symmetric

        # compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        labels = torch.cat([labels, labels], dim=0).contiguous().view(-1, 1)

        # compute the mask
        pos_mask = torch.eq(labels, labels.T).float().to(z_i.device)
        neg_mask = 1 - pos_mask
        pos_mask.fill_diagonal_(0)
        neg_mask.fill_diagonal_(0)

        similarity = torch.div(torch.matmul(z, z.T), self.temperature)
        # for numerical stability
        sim_row_max, _ = torch.max(similarity, dim=1, keepdim=True)
        similarity = similarity - sim_row_max.detach()

        log_prob = self.lamda * self._soft_max((similarity * neg_mask), axis=0)

        loss = log_prob

        loss = loss.view(2, batch_size).mean()

        return loss
