"""Loss functions only: BCE (via torch.nn), supervised contrastive loss,
the composite supervised loss with a learned contrastive weight, and the
SSL cross-modal regression objective. No model or data imports."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    """Supervised contrastive loss (SupCon / NT-Xent formulation).

    embeddings: (B, D) unnormalized embeddings.
    labels: (B,) binary class labels.
    Anchors with no positives in the batch contribute zero loss.
    """
    embeddings = F.normalize(embeddings.float(), dim=1)
    labels = labels.view(-1, 1)

    similarity = embeddings @ embeddings.t() / temperature
    similarity = similarity - similarity.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=embeddings.device)
    exp_sim = torch.exp(similarity).masked_fill(self_mask, 0.0)
    log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    positive_mask = (labels == labels.t()) & ~self_mask
    num_positives = positive_mask.sum(dim=1)
    has_positive = num_positives > 0

    if not has_positive.any():
        return torch.zeros((), device=embeddings.device, dtype=embeddings.dtype)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[has_positive] / num_positives[has_positive]
    return -mean_log_prob_pos.mean()


class CompositeSupervisedLoss(nn.Module):
    """L_sup = L_BCE + softplus(w) * L_contrastive, with w a learned
    nn.Parameter (non-negative via softplus)."""

    def __init__(self, temperature=0.1, init_logit=-2.5):
        super().__init__()
        self.temperature = temperature
        self.bce = nn.BCEWithLogitsLoss()
        self.contrastive_weight_logit = nn.Parameter(torch.tensor(float(init_logit)))

    def forward(self, logits, embeddings, labels):
        loss_bce = self.bce(logits.squeeze(1), labels)
        loss_contrastive = supervised_contrastive_loss(embeddings, labels, temperature=self.temperature)
        weight = F.softplus(self.contrastive_weight_logit)
        loss = loss_bce + weight * loss_contrastive
        components = {
            "bce": loss_bce.detach(),
            "contrastive": loss_contrastive.detach(),
            "contrastive_weight": weight.detach(),
        }
        return loss, components


class ProjectionMLP(nn.Module):
    """phi: 2-layer MLP mapping the FLAIR embedding into the QSM embedding
    space, used only during Stage 1 (SSL) pretraining."""

    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def ssl_regression_loss(pred_qsm, z_qsm):
    """L_SSL = || phi(z_F) - z_Q ||^2"""
    return F.mse_loss(pred_qsm, z_qsm)
