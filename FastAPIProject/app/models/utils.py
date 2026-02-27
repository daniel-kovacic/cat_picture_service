import json
import os
from typing import Type

import numpy as np
import torch
from numpy.typing import NDArray

from torch import nn


IMAGE_SIZE = (224, 224)
LANDMARK_COORD_SHAPE = (9, 2)

def load_best_model(model_path: str, model: Type[nn.Module], landmark_shape: tuple[int, int]) :
    model_filename = "model.pt"
    path = os.path.join(model_path, model_filename)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model = model(landmark_shape)
    model.load_state_dict(state_dict)
    return model

def load_model_dict(path: str) -> dict:
    path = os.path.join(path, "run.json")
    with open(path, "r") as f:
        return json.load(f)

def heatmaps_to_coords(hm: torch.Tensor):
    B, K, H, W = hm.shape
    hm_flat = hm.view(B, K, -1)
    idx = hm_flat.argmax(dim=2)

    y = idx // W
    x = idx % W
    coords = torch.stack([x, y], dim=2).float()
    return coords


def soft_argmax(hm):
    B, K, H, W = hm.shape
    hm = hm.view(B, K, -1)
    hm = torch.softmax(hm, dim=-1)
    xs = torch.linspace(0, W - 1, W).to(hm.device)
    ys = torch.linspace(0, H - 1, H).to(hm.device)
    xs, ys = torch.meshgrid(xs, ys, indexing="xy")
    coords_x = (hm * xs.flatten()).sum(-1)
    coords_y = (hm * ys.flatten()).sum(-1)
    return torch.stack([coords_x, coords_y], dim=-1)

def scale_landmarks_to_original(normalized_landmarks: NDArray,
                                 target_img_size: tuple[int, int],
                                 current_img_size: tuple[int, int]=(1,1)) -> NDArray:
    landmarks = np.zeros_like(normalized_landmarks)
    landmarks[:, 0] = normalized_landmarks[:, 0] * target_img_size[0]/current_img_size[0]
    landmarks[:, 1] = normalized_landmarks[:, 1] * target_img_size[1]/current_img_size[1]
    return landmarks
