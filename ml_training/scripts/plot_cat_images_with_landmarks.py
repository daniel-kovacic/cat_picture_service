import numpy as np
from PIL import Image
from torch import no_grad

from config import DATA_DIR, IMAGE_SHAPE
from data.dataset import CatLandmarkDataset
from data.utils import display_landmark_cat_image, display_landmark_cat_image_from_path, \
    load_raw_rgb_image, load_landmark_coord, get_landmark_coord_path, image_to_tensor, \
    normalized_coord_to_original_coord
from models.model import ResNetModel
from models.model_heatmap import ResNetHeatmap
from models.utils import load_best_model, load_model_dict

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def get_tensor(image: Image.Image):
    return image_to_tensor(image).unsqueeze(0)


def display_predictions(path: str, model: torch.nn.Module):
    image = load_raw_rgb_image(path)
    with torch.no_grad():
        pred_normalized = model(get_tensor(image)).to("cpu")
    pred_landmarks = normalized_coord_to_original_coord(pred_normalized, image.size)
    display_landmark_cat_image(image, pred_landmarks)


def show_heatmaps_with_true_coords(image: torch.Tensor,
                                   heatmaps: torch.Tensor,
                                   true_coords: torch.Tensor):
    K, H_hm, W_hm = heatmaps.shape
    C, H_img, W_img = image.shape

    heatmaps = F.interpolate(heatmaps.unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False).squeeze(
        0)

    if heatmaps.max() > 1.0:
        heatmaps = torch.sigmoid(heatmaps)

    img = image.permute(1, 2, 0).cpu()
    if C == 1:
        img = img.squeeze(2)

    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3))
    if K == 1:
        axes = [axes]

    for i in range(K):
        axes[i].imshow(img, cmap='gray')
        axes[i].imshow(heatmaps[i].cpu(), alpha=0.5, cmap='jet')
        x, y = true_coords[i]
        x = x * W_img / W_hm
        y = y * H_img / H_hm
        axes[i].scatter(x, y, c='lime', s=40, marker='x')
        axes[i].set_title(f'Landmark {i}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def plot_heatmap_predictions(path: str):
    if model_type == "landmark":
        display_landmark_cat_image_from_path(path)
        display_predictions(path, model)

    elif model_type == "heatmap":
        image = load_raw_rgb_image(path)
        image_tensor = get_tensor(image)
        landmark_path = get_landmark_coord_path(path)
        landmarks = load_landmark_coord(landmark_path)
        normalized_landmarks = np.zeros_like(landmarks)
        normalized_landmarks[:, 0] = (landmarks[:, 0] / image.size[0]) * IMAGE_SHAPE[0]
        normalized_landmarks[:, 1] = (landmarks[:, 1] / image.size[1]) * IMAGE_SHAPE[1]
        with no_grad():
            y_hat = model(image_tensor)
        show_heatmaps_with_true_coords(image_tensor.squeeze(0), y_hat.squeeze(0), normalized_landmarks)


if "__main__" == __name__:
    model_checkpoint_str = "20260218-172424"  # "20260217-180139"
    model_dict = load_model_dict(model_checkpoint_str)
    if "heatmap" in model_dict["model_name"]:
        model_type = "heatmap"
    else:
        model_type = "landmark"
    model = load_best_model(model_checkpoint_str, model=ResNetModel if model_type == "landmark" else ResNetHeatmap)

    val_dataset = CatLandmarkDataset(DATA_DIR, "val")
    train_dataset = CatLandmarkDataset(DATA_DIR, "train")

    for path in train_dataset.path_list[:1]:
        plot_heatmap_predictions(path)

    for path in val_dataset.path_list[:10]:
        plot_heatmap_predictions(path)
