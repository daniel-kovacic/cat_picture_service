from pathlib import Path
from typing import Optional
from typing import Literal

import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2

from config import IMAGE_SHAPE
from data.utils import load_raw_rgb_image
from models.model import ResNetModel
from models.model_heatmap import ResNetHeatmap
from models.peak_utils import PeakUtils
from models.utils import load_best_model
from numpy.typing import NDArray

RELATIVE_DIST_EYES = (1, 0)
RELATIVE_DIST_EYE_NOSE = (0.485, 0.805)
RELATIVE_PADDING_X = (0.7, 0.7)
RELATIVE_PADDING_Y = (1.5, 0.4)
IMAGE_SIZE = (224, 224)
PEAK_THRESHOLD = 0.52
MODEL_CHECKPOINT_NAME = "20260218-172424"


def sensible_landmarks(landmarks: NDArray) -> bool:
    if landmarks is None or len(landmarks) < 3:
        return False

    left_eye, right_eye, nose = landmarks[:3]

    if left_eye[0] >= right_eye[0]:
        return False

    eye_distance = np.linalg.norm(left_eye - right_eye)
    nose_left = np.linalg.norm(nose - left_eye)
    nose_right = np.linalg.norm(nose - right_eye)

    if eye_distance < 5:
        return False

    symmetry_ratio = nose_left / (nose_right + 1e-6)
    if symmetry_ratio < 0.5 or symmetry_ratio > 2.0:
        return False

    eye_y_mean = (left_eye[1] + right_eye[1]) / 2
    if nose[1] <= eye_y_mean:
        return False

    nose_eye_distance = (nose_left + nose_right) / 2
    distance_ratio = eye_distance / (nose_eye_distance + 1e-6)

    if distance_ratio < 0.7 or distance_ratio > 2.2:
        return False

    return True


def align_face(image: Image.Image, src_pts):
    image = np.array(image)
    dst_pts = np.array(get_ref_pts())
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    aligned = cv2.warpAffine(
        image,
        M,
        IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return Image.fromarray(aligned)


def _rescale_coord(value: float, axis: Literal[0, 1]):
    if axis == 0:
        size = RELATIVE_DIST_EYES[0] + sum(RELATIVE_PADDING_X)
    elif axis == 1:
        size = RELATIVE_DIST_EYE_NOSE[1] + sum(RELATIVE_PADDING_Y)
    else:
        raise Exception("axis should be 0 or 1")
    return value * (IMAGE_SIZE[axis] - 1) / (size - 1)


def get_ref_pts():
    left_eye_coord = np.array([_rescale_coord(RELATIVE_PADDING_X[0], 0),
                               _rescale_coord(RELATIVE_PADDING_Y[0], 1)])
    right_eye_coord = np.array([_rescale_coord(RELATIVE_PADDING_X[0] + RELATIVE_DIST_EYES[0], 0),
                                _rescale_coord(RELATIVE_PADDING_Y[0], 1)])
    nose_coord = np.array([_rescale_coord(RELATIVE_PADDING_X[0] + RELATIVE_DIST_EYE_NOSE[0], 0),
                           _rescale_coord(RELATIVE_PADDING_Y[0] + RELATIVE_DIST_EYE_NOSE[1], 1)])
    return left_eye_coord, right_eye_coord, nose_coord


def process_directory_tree(
        src_root: str | Path,
        dst_root: str | Path,
        max_dirs: Optional[int] = None,
) -> None:
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    model_type = "heatmap"
    model_checkpoint_str = MODEL_CHECKPOINT_NAME
    model = load_best_model(
        model_checkpoint_str,
        model=ResNetModel if model_type == "landmark" else ResNetHeatmap
    )
    subdirs = [p for p in src_root.iterdir() if p.is_dir()]
    subdirs.sort()

    if max_dirs is not None:
        subdirs = subdirs[:max_dirs]

    for src_dir in subdirs:
        dst_dir = dst_root / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        # counter = 0
        for item in src_dir.iterdir():
            if item.is_file():
                if not (item.name.endswith(".JPG") or item.name.endswith(".jpg")):
                    continue
                dst_file = dst_dir / item.name
                image = load_raw_rgb_image(item)
                rescaled_image = image.resize(IMAGE_SHAPE)
                image_tensor = T.ToTensor()(rescaled_image).unsqueeze(0)
                with torch.no_grad():
                    y_hat = model(image_tensor).squeeze(0)

                landmarks = PeakUtils.extract_cats_from_heatmap(
                    y_hat, peak_threshold=PEAK_THRESHOLD,
                    pool_kernel=15, max_peaks_per_channel=10, max_nose_to_landmark_scale=6
                )
                if len(landmarks) != 1:
                    # if counter % 20 == 0:
                    #     plt.imshow(image)
                    #     plt.title(f"{item.name} len = {len(landmarks)}")
                    #     plt.show()
                    # counter += 1
                    continue
                relevant_landmarks_np = (
                    landmarks[0][:3, :2]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                relevant_landmarks_np[:, 0] = relevant_landmarks_np[:, 0] * (image.size[0] - 1)
                relevant_landmarks_np[:, 1] = relevant_landmarks_np[:, 1] * (image.size[1] - 1)
                if not sensible_landmarks(relevant_landmarks_np):
                    continue
                image = align_face(image, relevant_landmarks_np)
                image.save(dst_file)


if __name__ == "__main__":
    process_directory_tree("../data/cat_individuals_dataset",
                           "../data/cat_individuals_dataset_preprocessed_v2",
                           max_dirs=None)
