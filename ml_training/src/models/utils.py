import json
import os

import torch

from config import MODEL_REGISTRY_PATH, IMAGE_SHAPE, LANDMARK_COORD_SHAPE
from models.model import ResNetModel
from training.trainer import ResNetModelTrainer


def load_best_model(path: str, model=ResNetModel):
    model_path = "model.pt"
    path = os.path.join(path, model_path)
    state_dict = torch.load(path, map_location=torch.device("cpu"))
    model = model(LANDMARK_COORD_SHAPE)
    model.load_state_dict(state_dict)
    return model


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


def load_model_dict(checkpoint_str: str) -> dict:
    path = os.path.join(MODEL_REGISTRY_PATH, checkpoint_str)
    path = os.path.join(path, "run.json")
    with open(path, "r") as f:
        return json.load(f)
