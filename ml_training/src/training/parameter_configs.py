from torch.optim import Adam

from config import LANDMARK_COORD_SHAPE, CAT_ID_CLASSES
from models.cat_identification_model import CombinedClassifier
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
parameter_configs["config_3"] = {
    "model": CombinedClassifier(CAT_ID_CLASSES, emb_dim=256, s=64.0, m=0.5, pretrained=True),
    "model_name": "resnet18_arc_face",
    "hyperparameters": {
        "lr": 5e-5,
        "batch_size": 64,
        "epochs": 5000,
        "optimizer": "adam",
        "emb_dim": 256,
        "s": 32,
        "m": 0.5,
        "pretrained": True,
    },
    "criterion": "mse",
}
