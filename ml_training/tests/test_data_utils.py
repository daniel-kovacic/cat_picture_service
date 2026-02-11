from pathlib import Path

import numpy as np
import pytest

from data.utils import get_cat_image_paths, get_landmark_coord_path, load_raw_rgb_image, load_landmark_coord


def test_get_cat_image_paths_finds_jpg_files(tmp_path):
    cat_image_rel_paths = ["pic1_.jpg", "pic2.jpg"]
    other_files = ["pic1_.jpg.cat", "pic1_.txt"]
    for rel_path in cat_image_rel_paths + other_files:
        (tmp_path / rel_path).touch()

    result = get_cat_image_paths(tmp_path)

    assert len(result) == len(cat_image_rel_paths)
    assert all(p.endswith(".jpg") for p in result)
    assert {Path(p).name for p in result} == set(cat_image_rel_paths)


def test_get_landmark_coord_path(tmp_path):
    cat_image_rel_paths = ["pic1_.jpg", "pic2.jpg"]
    other_files = ["pic1_.jpg.cat", "pic1_.txt"]

    correct_paths = []
    for rel_path in cat_image_rel_paths:
        correct_path = (tmp_path / (rel_path + ".cat"))
        correct_paths.append(str(tmp_path / rel_path))
        correct_path.touch()

    incorrect_paths = []
    for rel_path in other_files:
        (tmp_path / rel_path).touch()
        incorrect_paths.append(str(tmp_path / rel_path))

    for path in correct_paths:
        assert get_landmark_coord_path(path).endswith(".cat")

    for path in correct_paths:
        assert get_landmark_coord_path(path) == str(path) + ".cat"

    for path in incorrect_paths:
        with pytest.raises(FileNotFoundError):
            get_landmark_coord_path(path)


def test_load_raw_rgb_image(sample_image_path):
    image = load_raw_rgb_image(sample_image_path)
    with pytest.raises(FileNotFoundError):
        load_raw_rgb_image(sample_image_path / "some_image.jpg")
    assert image.size == (100, 200)


def test_load_landmark_coord(tmp_path):
    valid_coordinates = [str(i) for i in range(18)]
    valid_coordinate_tuples = np.array([np.array((i, i + 1)) for i in range(0, 18, 2)])
    valid_coord_list = ["9"] + valid_coordinates
    invalid_coordinate_lists = [valid_coordinates, ["8"] + valid_coordinates, valid_coord_list[:-1]]

    coord_tmp_files = []
    for i, coord_list in enumerate(invalid_coordinate_lists + [valid_coord_list]):
        file_path = tmp_path / f"{i}.jpg.cat"
        coord_tmp_files.append(str(file_path))
        coord_str = " ".join(coord_list)
        with file_path.open("w") as f:
            f.write(coord_str)

    for coord_file in coord_tmp_files[:-1]:
        with pytest.raises(Exception):
            load_landmark_coord(coord_file)
    assert (load_landmark_coord(coord_tmp_files[-1]) == valid_coordinate_tuples).all()
