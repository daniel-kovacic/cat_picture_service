from torch.optim import Adam

from config import LANDMARK_COORD_SHAPE
from models.model import ResNetModel
from models.model_heatmap import ResNetHeatmap

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

parameter_configs["config_2"] = {
    "model": ResNetHeatmap(LANDMARK_COORD_SHAPE),
    "model_name": "resnet18_heatmap",
    "hyperparameters": {
        "lr": 1e-3,
        "batch_size": 32,
        "epochs": 1000,
        "optimizer": "adam",
    },
    "criterion": "mse",
}
