import os

import torch
from PIL import Image

from core.config import LANDMARK_MODEL_PATH, FACE_ID_MODEL_PATH, NUMBER_OF_CLASSES
from core.util import get_device
from models.cat_identification_model import CombinedClassifier
from models.multi_cat_landmark_wrapper import MultiCatModelWrapper


class ModelSingleton:
    _instance = None
    @staticmethod
    def load_embedder_model(path:str):
        device = get_device()
        combined_classifer = CombinedClassifier(NUMBER_OF_CLASSES,
                                                pretrained=False)
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        combined_classifer.load_state_dict(state_dict)
        combined_classifer.eval()
        combined_classifer.to(device)
        return combined_classifer.embedder



    def __new__(cls,  *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = MultiCatModelWrapper(LANDMARK_MODEL_PATH)
            cls._instance.cat_embedder = ModelSingleton.load_embedder_model(os.path.join(FACE_ID_MODEL_PATH, "model.pt"))
        return cls._instance

    def __call__(self, image:Image.Image) -> torch.Tensor:
        return ModelSingleton._instance.model(image)


