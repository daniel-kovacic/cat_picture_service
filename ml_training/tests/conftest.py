from pathlib import Path
from PIL import Image

import pytest


@pytest.fixture
def sample_data_dir() -> Path:
    return Path(__file__).parent / "sample_data"


@pytest.fixture
def sample_image_path(tmp_path: Path):
    image_path = tmp_path / "image.jpg"
    test_image = Image.new("RGB", (100, 200))
    test_image.save(image_path)
    return image_path
