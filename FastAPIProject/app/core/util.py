from io import BytesIO

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


