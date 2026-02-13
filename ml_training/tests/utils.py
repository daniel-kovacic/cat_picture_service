import random
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
import torchvision.transforms as T

from config import IMAGE_SHAPE
from data.dataset import CatLandmarkDataset


def generate_random_datapoint():
    image = Image.new(mode="RGB", size=IMAGE_SHAPE)
    image_tensor = T.ToTensor()(image)
    image_tensor = image_tensor.reshape(1, 3, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    landmark_coords = [random.randint(0, 100) / 100 for _ in range(18)]
    landmark_tensor = torch.Tensor(landmark_coords).reshape(-1, 2)
    return image_tensor, landmark_tensor


def create_random_coordinates(width: int, height: int) -> list[int]:
    coordinates_x = random.sample(range(width), 9)
    coordinates_y = random.sample(range(height), 9)
    coordinates = [coord for coords in zip(coordinates_x, coordinates_y) for coord in coords]
    return coordinates


def create_valid_coord_str(coord_list: list[int]) -> str:
    coord_len_list = [9] + coord_list
    str_list = [str(x) for x in coord_len_list]
    return " ".join(str_list)


def create_valid_data_point_file(file_path: str, bitsize: tuple[int, int], coordinates: list[int] | None = None):
    image = Image.new(mode="RGB", size=tuple(bitsize))
    image.save(file_path)
    if coordinates is None:
        coordinates = create_random_coordinates(image.size[0], image.size[1])

    coord_str = create_valid_coord_str(coordinates)
    landmark_file_path = file_path + ".cat"
    landmark_path = Path(landmark_file_path)
    with landmark_path.open(mode="w") as f:
        f.write(coord_str)


def create_valid_data_point_files(path: Path, nmbr_of_files: int, base_file_path: str = "val_file_path"):
    file_paths = [path / f"{base_file_path}_{i}.jpg" for i in range(nmbr_of_files)]
    for p in file_paths:
        size = (random.randint(100, 1000), random.randint(100, 1000))
        coord = create_random_coordinates(*size)
        create_valid_data_point_file(str(p), size, coord)


def create_data_loader(path: Path,
                       split: Literal["train", "val", "test"] | list[Literal["train", "val", "test"]] = None):
    DataLoader = torch.utils.data.DataLoader(CatLandmarkDataset(path, split=split))
    return DataLoader
