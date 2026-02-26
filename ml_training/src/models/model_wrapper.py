import json
import os
from typing import Type

from PIL import Image
import torch
from torch import nn

from config import MODEL_REGISTRY_PATH, LANDMARK_COORD_SHAPE, IMAGE_SHAPE
from models.model import ResNetModel
from models.model_heatmap import ResNetHeatmap
import torchvision.transforms as T

from models.peak_utils import PeakUtils


class ModelWrapper:
    @staticmethod
    def _load_model_dict(checkpoint_str: str) -> dict:
        path = os.path.join(MODEL_REGISTRY_PATH, checkpoint_str)
        path = os.path.join(path, "run.json")
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def _load_best_model(checkpoint_str: str, model: Type[nn.Module]):
        model_filename = "model.pt"
        path = os.path.join(MODEL_REGISTRY_PATH, checkpoint_str)
        path = os.path.join(path, model_filename)
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        model = model(LANDMARK_COORD_SHAPE)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _get_model_type(model_type: str) -> Type[nn.Module]:
        if model_type == "resnet18":
            return ResNetModel
        elif model_type == "resnet18_heatmap":
            return ResNetHeatmap
        else:
            raise NotImplementedError(f"Loading of model type {model_type} is not implemented")

    def __init__(self, model_id: str):
        self.model_id = model_id
        path = os.path.join(MODEL_REGISTRY_PATH, model_id)
        self.model_dict = ModelWrapper._load_model_dict(path)
        self.model_type_str = self.model_dict["model_name"]
        self.model_type = ModelWrapper._get_model_type(self.model_type_str)
        self.model = self._load_best_model(model_id,
                                           model=self.model_type if self.model_type_str == "landmark" else ResNetHeatmap)
        self.model.eval()

    def __call__(self, image: Image.Image, output_multiple=False) -> torch.Tensor:
        rescaled_image = image.resize(IMAGE_SHAPE)
        image_tensor = T.ToTensor()(rescaled_image).unsqueeze(0)
        with torch.no_grad():
            y = self.model(image_tensor).squeeze(0)
        if self.model_type_str == "resnet18_heatmap":
            if output_multiple:
                return PeakUtils.find_multiple_most_likely_normalized_coord()
            rescaled_coords = ModelWrapper._find_most_likely_normalized_coord(y)
            normalized_landmarks = ModelWrapper._normalize_landmarks(rescaled_coords, IMAGE_SHAPE)

        elif self.model_type_str == "resnet18":
            normalized_landmarks = y
        else:
            raise NotImplementedError(f"model type {self.model_type_str} is not implemented")
        final_coords = ModelWrapper._scale_landmarks_to_original(normalized_landmarks, image.size)
        return final_coords

    @staticmethod
    def _scale_landmarks_to_original(normalized_landmarks: torch.Tensor,
                                     image_size: tuple[int, int]) -> torch.Tensor:
        landmarks = torch.zeros_like(normalized_landmarks)
        landmarks[:, 0] = normalized_landmarks[:, 0] * image_size[0]
        landmarks[:, 1] = normalized_landmarks[:, 1] * image_size[1]
        return landmarks

    @staticmethod
    def _find_most_likely_normalized_coord(hm) -> torch.Tensor:
        C, H, W = hm.shape

        flat_idx = torch.argmax(hm.view(C, -1), dim=-1)

        y = flat_idx // W
        x = flat_idx % W

        coords = torch.stack([x, y], dim=-1)
        return coords

    @staticmethod
    def _find_multiple_most_likely_normalized_coord(hm) -> torch.Tensor:
        C, H, W = hm.shape

        flat_idx = torch.argmax(hm.view(C, -1), dim=-1)

        y = flat_idx // W
        x = flat_idx % W

        coords = torch.stack([x, y], dim=-1)
        return coords

    @staticmethod
    def _normalize_landmarks(landmarks, image_size: tuple[int, int]) -> torch.Tensor:
        landmarks = landmarks.float()
        normalized_landmarks = torch.zeros_like(landmarks)
        normalized_landmarks[:, 0] = landmarks[:, 0] / image_size[0]
        normalized_landmarks[:, 1] = landmarks[:, 1] / image_size[1]
        return normalized_landmarks
