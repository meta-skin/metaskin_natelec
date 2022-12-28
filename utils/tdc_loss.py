
"""
TDC loss function

This code is generalization of NTXentLoss by giving tolerance to timely correlated sensor signals.

Given a unique index that represents relative distances between sensor signals, 
this code will calculate time discount factor proportional to the relative distance.

For the original NTXentLoss code, please refer to pytorch_metric_learning in "https://github.com/KevinMusgrave/pytorch-metric-learning"

"""


import torch
from pytorch_metric_learning.distances import CosineSimilarity


class TDCLoss(torch.nn.Module):
    def __init__(self, tol_dist, temperature=0.07):
        super().__init__()
        self.tol_dist = tol_dist
        self.temperature = temperature
        self.distance = CosineSimilarity()

    def forward(self, embeddings, labels):
        self.check_shapes(embeddings, labels)
        embeddings, labels = self.match_device(embeddings, labels)

        loss = self.compute_loss(embeddings, labels)
        return loss

    def check_shapes(self, embeddings, labels):
        if(labels is None):
            raise ValueError("labels cannot be None")
        if(embeddings.shape[0] != labels.shape[0]):
            raise ValueError("embeddings and labels must have the same length")

        if(embeddings.ndim != 2):
            raise ValueError("embeddings must be a 2D tensor")

        if(labels.ndim != 1):
            raise ValueError("labels must be a 1D tensor")

    def match_device(self, embeddings, labels):
        if(embeddings.device != labels.device):
            labels = labels.to(embeddings.device)

        return embeddings, labels

    def compute_loss(self, embeddings, labels):
        labels1 = labels.unsqueeze(1)
        labels2 = labels.unsqueeze(0)
        diff = torch.abs(labels1 - labels2)/self.tol_dist
        matches = diff < 1
        matches = matches.byte()
        negatives = matches ^ 1
        matches.fill_diagonal_(0)
        a1_idx, p_idx = torch.where(matches)
        a2_idx, n_idx = torch.where(negatives)

        indices_tuple = [a1_idx, p_idx, a2_idx, n_idx]
        if all(len(x) <= 1 for x in indices_tuple):
            return 0
        mat = self.distance(embeddings)
        return self.pair_based_loss(mat, indices_tuple, diff)

    def pair_based_loss(self, mat, indices_tuple, diff):
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair, pos_pair_diff = [], [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
            pos_pair_diff = diff[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]

        return self._compute_loss(pos_pair, neg_pair, indices_tuple, pos_pair_diff)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple, pos_pair_diff):
        a1, p, a2, _ = indices_tuple
        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            n_per_p = a2.unsqueeze(0) == a1.unsqueeze(1)
            neg_pairs = neg_pairs * n_per_p

            neg_pairs[n_per_p == 0] = torch.finfo(dtype).min
            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)

            time_discount_factor = torch.exp(-torch.pow(2*pos_pair_diff, 2))
            numerator = numerator * time_discount_factor

            denominator = torch.sum(
                torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log(
                (numerator / denominator) + torch.finfo(dtype).tiny)
            return torch.mean(-log_exp)

        return 0



