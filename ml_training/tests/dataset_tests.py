import random

import numpy as np
import pytest
import torch

from config import LANDMARK_COORD_SHAPE
from data.dataset import CatLandmarkDataset
from data.utils import normalize_landmark_coordinates
from utils import create_valid_data_point_files, create_random_coordinates


class TestCatLandmarkDataset:

    @pytest.fixture
    def dataset(self, tmp_path):
        valid_paths_0 = [f"valid_0_{i}.jpg" for i in range(10)]
        valid_paths_1 = [f"valid_1_{i}.jpg" for i in range(10)]
        sizes_img = [(random.randint(100, 500), random.randint(100, 500)) for _ in range(10)]
        valid_paths = create_random_coordinates(100, 500)
        invalid_coordinates = []

        invalid_coordinates.append(valid_paths[:17])
        invalid_coordinates.append(valid_paths + [50])
        invalid_coordinates.append(valid_paths[:17] + [-1])
        invalid_coordinates.append(valid_paths[:17] + [1000])
        invalid_coordinates.append([-1] + valid_paths[:17])
        invalid_coordinates.append([1000] + valid_paths[:17])

        for path, size in zip(valid_paths_0, sizes_img):
            create_valid_data_point_files(str(tmp_path / path), size)

        for path, size, coord in zip(valid_paths_1[:len(invalid_coordinates)],
                                     sizes_img[:len(invalid_coordinates)],
                                     invalid_coordinates):
            create_valid_data_point_files(str(tmp_path / path), size, coord)

        return CatLandmarkDataset(str(tmp_path))

    def test_len(self, dataset: CatLandmarkDataset):
        assert len(dataset) == 10

    def test_getitem(self, dataset: CatLandmarkDataset):
        for i in range(10):
            x, y = dataset[i]
            assert type(x) == torch.Tensor
            assert type(y) == torch.Tensor
            assert x.ndim == 3
            assert x.shape[0] == 3
            assert y.ndim == 2
            assert y.shape == LANDMARK_COORD_SHAPE

        with pytest.raises(IndexError):
            dataset[11]
