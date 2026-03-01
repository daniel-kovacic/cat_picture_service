from typing import Type

from numpy.typing import NDArray
import numpy as np
from PIL import Image
import torch
from torch import nn

from core.util import get_device
from models.model_heatmap import ResNetHeatmap
import torchvision.transforms as T

from models.peak_utils import PeakUtils
from models.utils import load_best_model, load_model_dict, LANDMARK_COORD_SHAPE, IMAGE_SIZE, scale_landmarks_to_original


class MultiCatModelWrapper:

    @staticmethod
    def _get_model_type(model_type: str) -> Type[nn.Module]:
        if model_type != "resnet18_heatmap":
            raise RuntimeError("Wrong model type")
        return ResNetHeatmap

    def __init__(self, model_path: str, peak_threshold: float = 0.52):
        self.model_dict = load_model_dict(model_path)
        self.model_type = MultiCatModelWrapper._get_model_type(self.model_dict["model_name"])
        self.model = load_best_model(model_path, ResNetHeatmap, LANDMARK_COORD_SHAPE)
        self.model.to(get_device())
        self.model.eval()
        self.peak_threshold = peak_threshold



    def __call__(self, image: Image.Image) -> list[NDArray]:
        rescaled_image = image.resize(IMAGE_SIZE)
        image_tensor = T.ToTensor()(rescaled_image).unsqueeze(0).to(get_device())
        with torch.no_grad():
            y_hat = self.model(image_tensor).squeeze(0)
        landmarks = PeakUtils.extract_cats_from_heatmap(
            y_hat, peak_threshold=self.peak_threshold,
            pool_kernel=15, max_peaks_per_channel=10, max_nose_to_landmark_scale=2.5
        )
        landmarks = [landmark[:,:2] for landmark in landmarks]
        landmarks_np = [landmark.cpu().numpy().astype(np.float32) for landmark in landmarks]
        final_coords = [scale_landmarks_to_original(landmark, image.size)
                        for landmark in landmarks_np]
        return final_coords




