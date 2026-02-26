import os
import random
from typing import Literal

import torch
from torch.utils.data import Dataset

from data.utils import get_cat_image_paths, check_data_integrity, load_raw_rgb_image, get_landmark_coord_path, \
    normalize_landmark_coordinates, load_landmark_coord, image_to_tensor, split_dataset

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15


class CatLandmarkDataset(Dataset):

    def _remove_corrupted_data(self):
        self.path_list = [p for p in self.path_list if check_data_integrity(p)]

    def __init__(self,
                 data_dir: str | os.PathLike,
                 split: Literal["train", "val", "test"] | list[Literal["train", "val", "test"]] = None,
                 seed=42):
        self.path_list = get_cat_image_paths(data_dir)
        self.seed = seed
        self.split = split
        if self.split is None:
            self.split = ["train", "val", "test"]

        self._remove_corrupted_data()

        random.seed(self.seed)
        random.shuffle(self.path_list)

        self.path_list = split_dataset(self.path_list, self.split, TRAIN_RATIO, VALIDATION_RATIO)

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path_str = self.path_list[idx]
        landmark_path = get_landmark_coord_path(path_str)

        raw_image = load_raw_rgb_image(path_str)
        image_tensor = image_to_tensor(raw_image)

        landmark_array = load_landmark_coord(landmark_path)
        norm_landmark_array = normalize_landmark_coordinates(landmark_array, *raw_image.size)

        return image_tensor, torch.from_numpy(norm_landmark_array)
