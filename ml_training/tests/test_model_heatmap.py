from config import LANDMARK_COORD_SHAPE, IMAGE_SHAPE
from models.model import ResNetModel
from models.model_heatmap import ResNetHeatmap
from utils import generate_random_datapoint


def test_model():
    x, y = generate_random_datapoint()
    model = ResNetHeatmap(y.shape)
    y0 = model(x)
    assert y0.shape == (1, LANDMARK_COORD_SHAPE[0], *IMAGE_SHAPE)
