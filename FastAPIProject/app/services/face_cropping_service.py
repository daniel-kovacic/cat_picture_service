import uuid

import PIL
import matplotlib
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, FastAPI
from PIL import Image
from torch import Tensor

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from core.util import read_image, fig_to_buf, image_to_buffer
from models.model_singleton import ModelSingleton
from fastapi.responses import StreamingResponse
from fastapi import Request
router = APIRouter()

RELATIVE_PADDING_X = (0.05, 0.05)
RELATIVE_PADDING_Y = (0.05, 0.2)
cropped_store = {}

def _crop_image(image: PIL.Image.Image, x_boundaries:tuple[float, float], y_boundaries: tuple[float, float]):
    box = (x_boundaries[0], y_boundaries[0], x_boundaries[1], y_boundaries[1])
    cropped = image.crop(box)
    return cropped

@router.post("/crop")
def cat_annotation(request:Request, file:UploadFile = File(...)):
    image = read_image(file)
    landmarks = ModelSingleton()(image)
    boundary_list = [calculate_boundaries(landmark) for landmark in landmarks]

    cropped = [_crop_image(image, boundaries_x, boundaries_y) for boundaries_x, boundaries_y in boundary_list]

    results = []
    for cropped in cropped:
        uid = str(uuid.uuid4())
        cropped_store[uid] = cropped
        results.append(str(request.url_for("get_cropped", cropped_id=uid)))
    return {"images": results}


def calculate_boundaries(landmark: np.typing.NDArray) -> tuple[tuple[float, float], tuple[float, float]]:
    print(landmark)
    landmark = landmark[~np.isnan(landmark).any(axis=1)]
    print(landmark)
    max_x = landmark[:, 0].max().item()
    min_x = landmark[:, 0].min().item()
    dist_x = max_x - min_x

    max_y = landmark[:, 1].max().item()
    min_y = landmark[:, 1].min().item()
    dist_y = max_y - min_y

    boundaries_x = (min_x - dist_x * RELATIVE_PADDING_X[0], max_x + dist_x * RELATIVE_PADDING_X[1])
    boundaries_y = (min_y - dist_y * RELATIVE_PADDING_Y[0], max_y + dist_y * RELATIVE_PADDING_Y[1])
    return boundaries_x, boundaries_y


@router.get("/crop/{cropped_id}", name="get_cropped")
def get_aligned_face(cropped_id):
    aligned_cat_face = cropped_store[cropped_id]

    return StreamingResponse(image_to_buffer(aligned_cat_face), media_type="image/png")