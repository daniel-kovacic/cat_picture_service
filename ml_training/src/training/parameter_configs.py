from torch.optim import Adam

from config import LANDMARK_COORD_SHAPE
from models.model import ResNetModel

parameter_configs = {}
parameter_configs["config_1"] = {
    "model": ResNetModel(LANDMARK_COORD_SHAPE),
    "model_name": "resnet18",
    "hyperparameters": {
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 100,
        "optimizer": "adam",
    },
    "criterion": "mse",
}
