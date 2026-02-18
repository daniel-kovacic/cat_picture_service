import os

import torch

from config import MODEL_REGISTRY_PATH, IMAGE_SHAPE, LANDMARK_COORD_SHAPE
from models.model import ResNetModel
from training.trainer import ResNetModelTrainer


def load_best_model(checkpoint_str: str):
    model_path = "model.pt"
    path = os.path.join(MODEL_REGISTRY_PATH, checkpoint_str)
    path = os.path.join(path, model_path)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model = ResNetModel(LANDMARK_COORD_SHAPE)
    model.load_state_dict(state_dict)
    return model
