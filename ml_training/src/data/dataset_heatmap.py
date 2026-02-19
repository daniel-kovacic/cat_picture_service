import os
import random
from typing import Literal

import torch
from torch.utils.data import Dataset

from config import IMAGE_SHAPE
from data.dataset import CatLandmarkDataset
from data.utils import get_cat_image_paths, check_data_integrity, load_raw_rgb_image, get_landmark_coord_path, \
    normalize_landmark_coordinates, load_landmark_coord, rescale_image, image_to_tensor

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15


class CatLandmarkHeatmapDataset(Dataset):

    def __init__(self,
                 data_dir: str | os.PathLike,
                 split: Literal["train", "val", "test"] | list[Literal["train", "val", "test"]] = None,
                 seed=42,
                 sigma=5
                 ):
        self.dataset = CatLandmarkDataset(data_dir, split, seed=seed)
        self.sigma = sigma

    def __len__(self) -> int:
        return len(self.dataset)

    def create_gaussian_heatmaps(self, landmarks, H, W):
        K = landmarks.shape[0]

        y = torch.arange(0, H).view(H, 1)
        x = torch.arange(0, W).view(1, W)

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        x0 = landmarks[:, 0].view(K, 1, 1)
        y0 = landmarks[:, 1].view(K, 1, 1)

        heatmaps = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

        return heatmaps

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[idx]
        y[:, 0] = y[:, 0] * IMAGE_SHAPE[0]
        y[:, 1] = y[:, 1] * IMAGE_SHAPE[1]
        y_heatmap = self.create_gaussian_heatmaps(y, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
        return x, y_heatmap
