from typing import Any

import PIL
import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, FastAPI
from matplotlib import pyplot as plt

from core.config import SIMILARITY_THRESHOLD
from core.face_realignment import align_face
from core.util import read_image, fig_to_buf, get_device
from models.model_singleton import ModelSingleton
import torchvision.transforms as T
import torch.nn.functional as F

router = APIRouter()


@router.post("/similarity")
def calculate_similarity(image_1:UploadFile = File(...), image_2: UploadFile = File(...)) -> dict[str, Any]:
    image1 = read_image(image_1)
    image2 = read_image(image_2)
    landmark_model = ModelSingleton().model
    embedding_model = ModelSingleton().cat_embedder
    with torch.no_grad():
        landmarks_1 = landmark_model(image1)
        landmarks_2 = landmark_model(image2)

    if len(landmarks_1) != 1 and len(landmarks_2) != 1:
        raise Exception("Require exactly one cat per image")
    landmark_1  =  landmarks_1[0].astype(np.float32)[:3]
    landmark_2 = landmarks_2[0].astype(np.float32)[:3]
    aligned_face1 = align_face(image1, landmark_1)
    aligned_face2 = align_face(image2, landmark_2)
    aligned_image_1 = T.ToTensor()(aligned_face1).unsqueeze(0).to(get_device())
    aligned_image_2 = T.ToTensor()(aligned_face2).unsqueeze(0).to(get_device())
    with torch.no_grad():
        emb1 = embedding_model(aligned_image_1)
        emb2 = embedding_model(aligned_image_2)
    sim = F.cosine_similarity(emb1, emb2).item()

    return {"similarity": sim,
            "same_cat": sim>SIMILARITY_THRESHOLD,
            "threshold": SIMILARITY_THRESHOLD}


