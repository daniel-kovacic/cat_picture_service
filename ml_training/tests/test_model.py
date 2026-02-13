from models.model import ResNetModel
from utils import generate_random_datapoint


def test_model():
    x, y = generate_random_datapoint()
    model = ResNetModel(y.shape)
    y0 = model(x)
    assert y0.shape == y.unsqueeze(0).shape
    assert (y0 >= 0).all()
    assert (y0 <= 1).all()
