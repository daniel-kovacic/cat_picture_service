import os
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def get_cat_image_paths(parent_dir: str | os.PathLike) -> list[str]:
    if not os.path.isdir(parent_dir):
        raise Exception(f"Parent dir {parent_dir} is not a directory")
    paths = [str(p) for p in Path(parent_dir).glob("**/*.jpg")]
    return paths


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
        print(split_line)
        coordinate_list = [int(c) for c in split_line]
        if len(coordinate_list) != 19 or coordinate_list[0] != 9:
            raise Exception(f"Landmark path {path} is not formatted as expected")
        coordinates = np.array(
            [np.array([coordinate_list[i], coordinate_list[i + 1]]) for i in range(1, len(coordinate_list), 2)])
        return coordinates


def display_landmark_cat_image(image_path: str | os.PathLike):
    landmark_path = get_landmark_coord_path(image_path)
    landmarks = load_landmark_coord(landmark_path)
    cat_image = load_raw_rgb_image(image_path)
    plt.imshow(cat_image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()


def check_data_integrity(path: str | os.PathLike) -> bool:
    try:
        load_raw_rgb_image(path)
        load_landmark_coord(path)
        return True
    except:
        return False
