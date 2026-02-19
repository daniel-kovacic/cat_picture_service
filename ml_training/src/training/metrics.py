import torch
from torch.nn import MSELoss


def avg_dist(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    per_landmark_dist = torch.linalg.norm(pred - target, dim=2)

    per_sample_avg = per_landmark_dist.mean(dim=1)

    return per_sample_avg.mean()


def mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return MSELoss()(pred, target)
