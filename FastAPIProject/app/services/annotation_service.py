import PIL
import matplotlib
import torch
from fastapi import APIRouter, UploadFile, File, FastAPI
from PIL import Image
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from core.util import read_image, fig_to_buf
from models.model_singleton import ModelSingleton
from fastapi.responses import StreamingResponse

router = APIRouter()

def _annotate_image(image: PIL.Image.Image, landmarks:torch.Tensor|list[torch.Tensor]):
    fig, ax = plt.subplots()
    ax.imshow(image)
    if type(landmarks) is torch.Tensor:
        landmarks = [landmarks]
    for landmark in landmarks:
        ax.scatter(landmark[:, 0], landmark[:, 1])
    ax.axis("off")
    return fig

@router.post("/annotate")
def cat_annotation(file:UploadFile = File(...)):
    image = read_image(file)
    landmark = ModelSingleton()(image)
    fig = _annotate_image(image, landmark)
    return StreamingResponse( fig_to_buf(fig), media_type="image/png")


