import os
import random
from typing import Literal

import torch
from torch.utils.data import Dataset

from data.utils import load_raw_rgb_image, image_to_tensor, \
    get_labeled_individual_cat_paths, split_dataset

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15


class CatIdDataset(Dataset):

    def __init__(self,
                 data_dir: str | os.PathLike,
                 split: Literal["train", "val", "test"] | list[Literal["train", "val", "test"]] = None,
                 seed=42):
        self.path_list = get_labeled_individual_cat_paths(data_dir)

        self.seed = seed

        self.split = split
        if self.split is None:
            self.split = ["train", "val", "test"]

        random.seed(self.seed)
        random.shuffle(self.path_list)

        self.path_list = split_dataset(self.path_list, self.split, TRAIN_RATIO, VALIDATION_RATIO)

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path_str, label = self.path_list[idx]

        raw_image = load_raw_rgb_image(path_str)
        image_tensor = image_to_tensor(raw_image)

        return image_tensor, label
