import torch
import torch.nn as nn

class NTXentLoss(torch.nn.Module):
    """
    A robust implementation of the NT-Xent loss from SimCLR.
    This version dynamically handles varying batch sizes.
    """
    def __init__(self, temperature, device):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def _get_correlated_mask(self, batch_size):
        """Creates a mask to select negative samples for a given batch size."""
        N = 2 * batch_size
        mask = torch.ones((N, N), device=self.device, dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Calculates the NT-Xent loss for a batch of embeddings.
        z_i and z_j are two different augmentations of the same batch of samples.
        """
        # Get the batch size dynamically from the input tensor
        batch_size = z_i.shape[0]
        if batch_size == 0:
            return torch.tensor(0.0, device=self.device)

        N = 2 * batch_size

        # Concatenate the two views of the batch
        z = torch.cat((z_i, z_j), dim=0)

        # Calculate the cosine similarity matrix
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        # Get the positive and negative samples
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)

        # Get the mask for negative samples dynamically
        negative_mask = self._get_correlated_mask(batch_size)
        negative_samples = sim[negative_mask].view(N, -1)

        # The labels are always 0 because the positive sample is the first column
        labels = torch.zeros(N, device=self.device).long()

        # Combine positive and negative samples to form logits
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        
        loss = self.criterion(logits, labels)
        loss /= N # Average the loss over all samples in the batch

        return loss