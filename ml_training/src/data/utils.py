import os
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import torchvision.transforms as T

from config import LANDMARK_COORD_SHAPE, IMAGE_SHAPE


def get_cat_image_paths(parent_dir: str | os.PathLike) -> list[str]:
    if not os.path.isdir(parent_dir):
        raise Exception(f"Parent dir {parent_dir} is not a directory")
    paths = [str(p) for p in Path(parent_dir).glob("**/*.jpg")]
    return paths


def get_labeled_individual_cat_paths(parent_dir: str | os.PathLike) -> list[tuple[str, int]]:
    if not os.path.isdir(parent_dir):
        raise Exception(f"Parent dir {parent_dir} is not a directory")
    root_dir = Path(parent_dir)
    subdirs = [d for d in root_dir.iterdir() if d.is_dir()]
    print(subdirs)
    subdirs_sorted = sorted(subdirs, key=lambda d: d.name)
    directory_label_dict = {p: l for l, p in enumerate(subdirs_sorted)}
    labeled_data = [(str(p), l) for dir_path, l in directory_label_dict.items() for p in dir_path.iterdir() if
                    p.is_file()]
    return labeled_data


def get_landmark_coord_path(image_path: str | os.PathLike) -> str:
    landmark_path = Path(image_path + ".cat")
    if not landmark_path.exists():
        raise FileNotFoundError(f"Landmark path {landmark_path} does not exist")
    return str(landmark_path)


def load_raw_rgb_image(path: str | os.PathLike) -> Image.Image:
    return Image.open(path)


def load_landmark_coord(path: str | os.PathLike) -> NDArray[np.int_]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Landmark path {path} does not exist")
    with open(path, "r") as f:
        line = f.readline()
        split_line = line.strip().split(" ")
        coordinate_list = [int(c) for c in split_line]
        if len(coordinate_list) != 19 or coordinate_list[0] != 9:
            raise Exception(f"Landmark path {path} is not formatted as expected")
        coordinates = np.array(
            [np.array([coordinate_list[i], coordinate_list[i + 1]]) for i in range(1, len(coordinate_list), 2)])
        return coordinates


def display_landmark_cat_image_from_path(image_path: str | os.PathLike):
    landmark_path = get_landmark_coord_path(image_path)
    landmarks = load_landmark_coord(landmark_path)
    cat_image = load_raw_rgb_image(image_path)
    display_landmark_cat_image(cat_image, landmarks)


def display_landmark_cat_image(cat_image: Image.Image, landmarks: np.ndarray):
    plt.imshow(cat_image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()


def normalize_landmark_coordinates(coord: NDArray[np.int_], size_x: int, size_y: int) -> NDArray[np.floating]:
    if coord.shape != LANDMARK_COORD_SHAPE:
        raise ValueError(f"Landmark coord shape {coord} is not formatted as expected")
    normalized_coord = np.zeros(LANDMARK_COORD_SHAPE)
    normalized_coord[:, 0] = coord[:, 0] / size_x
    normalized_coord[:, 1] = coord[:, 1] / size_y
    return normalized_coord.astype(np.float32)


def rescale_image(image: Image.Image) -> Image.Image:
    return image.resize(IMAGE_SHAPE)


def check_data_integrity(path: str | os.PathLike) -> bool:
    try:
        image = load_raw_rgb_image(path)
        coord = load_landmark_coord(get_landmark_coord_path(path))
        size_x, size_y = image.size
        if any(x < 0 or y < 0 or x > size_x or y > size_y for x, y in coord):
            raise ValueError("Invalid coordinates")
        return True
    except Exception as e:
        return False


def image_to_tensor(raw_image: Image.Image) -> torch.Tensor:
    rescaled_image = rescale_image(raw_image)
    image_tensor = T.ToTensor()(rescaled_image)
    return image_tensor


def normalized_coord_to_original_coord(coord, image_shape) -> NDArray[np.floating]:
    rescaled_coords = np.zeros(LANDMARK_COORD_SHAPE)
    rescaled_coords[:, 0] = coord[0, :, 0] * image_shape[0]
    rescaled_coords[:, 1] = coord[0, :, 1] * image_shape[1]
    return rescaled_coords


def _get_split_indices(data_size: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    return int(data_size * train_ratio), int(data_size * (train_ratio + val_ratio))


def split_dataset(data_list: list,
                  split: Literal["train", "val", "test"] | list[Literal["train", "val", "test"]],
                  train_ratio: float,
                  val_ratio: float,
                  ):
    val_idx, test_idx = _get_split_indices(len(data_list), train_ratio, val_ratio)
    index_dict = {'train': (None, val_idx), 'val': (val_idx, test_idx), 'test': (test_idx, None)}
    splits = split if type(split) != str else [split]
    data = []
    for split in splits:
        if not split in index_dict:
            raise ValueError(f'Split {split} does not exist')
        data.extend(data_list[index_dict[split][0]:index_dict[split][1]])
    return data
