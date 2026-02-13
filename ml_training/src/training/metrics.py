import torch
import numpy as np
from numpy.typing import NDArray

from config import LANDMARK_COORD_SHAPE


def avg_dist(pred: NDArray, target: NDArray):
    sum_dist = np.linalg.norm(pred - target, axis=2).sum()
    avg_dist = sum_dist / LANDMARK_COORD_SHAPE[0]
    return avg_dist


def mse(pred: NDArray, target: NDArray):
    return np.mean((pred - target) ** 2)
