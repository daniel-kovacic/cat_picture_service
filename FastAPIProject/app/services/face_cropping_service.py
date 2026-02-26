import PIL
import matplotlib
import torch
from fastapi import APIRouter, UploadFile, File, FastAPI
from PIL import Image
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from core.util import read_image, fig_to_buf, image_to_buffer
from models.model_singleton import ModelSingleton
from fastapi.responses import StreamingResponse

router = APIRouter()

RELATIVE_PADDING_X = (0.05, 0.05)
RELATIVE_PADDING_Y = (0.05, 0.2)

def _crop_image(image: PIL.Image.Image, x_boundaries:tuple[float, float], y_boundaries: tuple[float, float]):
    box = (x_boundaries[0], y_boundaries[0], x_boundaries[1], y_boundaries[1])
    cropped = image.crop(box)
    return cropped

@router.post("/crop")
def cat_annotation(file:UploadFile = File(...)):
    image = read_image(file)
    landmark = ModelSingleton()(image)
    max_x = landmark[:, 0].max().item()
    min_x = landmark[:, 0].min().item()
    dist_x = max_x - min_x

    max_y = landmark[:, 1].max().item()
    min_y = landmark[:, 1].min().item()
    dist_y = max_y - min_y

    boundaries_x = (min_x - dist_x*RELATIVE_PADDING_X[0], max_x + dist_x*RELATIVE_PADDING_X[1])
    boundaries_y = (min_y - dist_y*RELATIVE_PADDING_Y[0], max_y  + dist_y*RELATIVE_PADDING_Y[1])

    cropped = _crop_image(image, boundaries_x, boundaries_y)


    return StreamingResponse( image_to_buffer(cropped), media_type="image/png")


