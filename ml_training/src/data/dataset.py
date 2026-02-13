import os
import random
from typing import Literal

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from data.utils import get_cat_image_paths, check_data_integrity, load_raw_rgb_image, get_landmark_coord_path, \
    normalize_landmark_coordinates, load_landmark_coord, rescale_image

TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15


class CatLandmarkDataset(Dataset):

    def _remove_corrupted_data(self):
        self.path_list = [p for p in self.path_list if check_data_integrity(p)]
        print(self.path_list)

    def _shuffle_data(self):
        random.shuffle(self.path_list)

    def _get_split_indices(self) -> tuple[int, int]:
        l = len(self.path_list)
        return int(l * TRAIN_RATIO), int(l * (TRAIN_RATIO + VALIDATION_RATIO))

    def _split_dataset(self):
        if self.split is None:
            self.split = ["train", "val", "test"]
        val_idx, test_idx = self._get_split_indices()
        index_dict = {'train': (None, val_idx), 'val': (val_idx, test_idx), 'test': (test_idx, None)}
        splits = self.split if type(self.split) != str else [self.split]
        data = []
        for split in splits:
            if not split in index_dict:
                raise ValueError(f'Split {split} does not exist')
            data.extend(self.path_list[index_dict[split][0]:index_dict[split][1]])
        self.path_list = data

    def __init__(self,
                 data_dir: str | os.PathLike,
                 split: Literal["train", "val", "test"] | list[Literal["train", "val", "test"]] = None,
                 seed=42):
        self.path_list = get_cat_image_paths(data_dir)
        self.seed = seed
        self.split = split

        random.seed(self.seed)
        self._remove_corrupted_data()

        self._shuffle_data()
        self._split_dataset()
        print(len(self.path_list))

    def __len__(self) -> int:
        return len(self.path_list)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path_str = self.path_list[idx]
        landmark_path = get_landmark_coord_path(path_str)

        raw_image = load_raw_rgb_image(path_str)
        rescaled_image = rescale_image(raw_image)
        image_tensor = T.ToTensor()(rescaled_image)

        landmark_array = load_landmark_coord(landmark_path)
        norm_landmark_array = normalize_landmark_coordinates(landmark_array, *raw_image.size)

        return image_tensor, torch.from_numpy(norm_landmark_array)
