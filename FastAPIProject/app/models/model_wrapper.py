from pathlib import Path
from typing import Type

from PIL import Image
import torch
from torch import nn

from models.model import ResNetModel
from models.model_heatmap import ResNetHeatmap
import torchvision.transforms as T

from models.utils import load_model_dict, LANDMARK_COORD_SHAPE, IMAGE_SIZE, load_best_model, scale_landmarks_to_original


class ModelWrapper:

    @staticmethod
    def _get_model_type(model_type: str) -> Type[nn.Module]:
        if model_type == "resnet18":
            return ResNetModel
        elif model_type == "resnet18_heatmap":
            return ResNetHeatmap
        else:
            raise NotImplementedError(f"Loading of model type {model_type} is not implemented")

    def __init__(self, model_path: str):
        self.model_dict = load_model_dict(model_path)
        self.model_type_str = self.model_dict["model_name"]
        self.model_type = ModelWrapper._get_model_type(self.model_type_str)
        self.model = load_best_model(model_path,
                      self.model_type if self.model_type_str == "landmark" else ResNetHeatmap,
                      LANDMARK_COORD_SHAPE)
        self.model.eval()

    def __call__(self, image: Image.Image) -> torch.Tensor:
        rescaled_image = image.resize(IMAGE_SIZE)
        image_tensor = T.ToTensor()(rescaled_image).unsqueeze(0)
        with torch.no_grad():
            y = self.model(image_tensor).squeeze(0)
        if self.model_type_str == "resnet18_heatmap":
            rescaled_coords = ModelWrapper._find_most_likely_normalized_coord(y)
            normalized_landmarks = ModelWrapper._normalize_landmarks(rescaled_coords, IMAGE_SIZE)
        elif self.model_type_str == "resnet18":
            normalized_landmarks = y
        else:
            raise NotImplementedError(f"model type {self.model_type_str} is not implemented")
        final_coords = scale_landmarks_to_original(normalized_landmarks, image.size)
        return final_coords



    @staticmethod
    def _find_most_likely_normalized_coord(hm) -> torch.Tensor:
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
