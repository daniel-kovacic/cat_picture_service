import random
from pathlib import Path

from PIL import Image


def create_random_coordinates(width: int, height: int) -> list[int]:
    coordinates_x = random.sample(range(width), 9)
    coordinates_y = random.sample(range(height), 9)
    coordinates = [coord for coords in zip(coordinates_x, coordinates_y) for coord in coords]
    return coordinates


def create_valid_coord_str(coord_list: list[int]) -> str:
    coord_len_list = [9] + coord_list
    str_list = [str(x) for x in coord_len_list]
    return " ".join(str_list)


def create_valid_data_point_files(file_path: str, bitsize: tuple[int, int], coordinates: list[int] | None = None):
    image = Image.new(mode="RGB", size=tuple(bitsize))
    image.save(file_path)
    if coordinates is None:
        coordinates = create_random_coordinates(image.size[0], image.size[1])

    coord_str = create_valid_coord_str(coordinates)
    landmark_file_path = file_path + ".cat"
    landmark_path = Path(landmark_file_path)
    with landmark_path.open(mode="w") as f:
        f.write(coord_str)
