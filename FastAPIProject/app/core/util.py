from io import BytesIO

import torch
from PIL import Image
from fastapi import UploadFile, File
from matplotlib import pyplot as plt
import matplotlib.figure

def fig_to_buf(fig: matplotlib.figure.Figure):
    buf = BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return buf

def image_to_buffer(image: Image.Image) -> BytesIO:
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return buf

def read_image(file:UploadFile = File(...)):
    image_bytes = file.file.read()
    return Image.open(BytesIO(image_bytes))


def get_device():
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    return torch.device(device_str)