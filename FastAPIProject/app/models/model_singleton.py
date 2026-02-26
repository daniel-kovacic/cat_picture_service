import torch
from PIL import Image

from core.config import LANDMARK_MODEL_PATH
from models.model_wrapper import ModelWrapper


class ModelSingleton:
    _instance = None

    def __new__(cls,  *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = ModelWrapper(LANDMARK_MODEL_PATH)
        return cls._instance

    def __call__(self, image:Image.Image) -> torch.Tensor:
        return ModelSingleton._instance.model(image)


