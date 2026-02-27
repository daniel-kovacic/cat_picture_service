from fastapi import APIRouter, UploadFile, File, FastAPI

from core.face_realignment import align_face
import uuid


from core.util import read_image, fig_to_buf, image_to_buffer
from models.model_singleton import ModelSingleton
from fastapi.responses import StreamingResponse
from fastapi import Request

router = APIRouter()
face_store = {}
@router.post("/align_face")
def cat_annotation(request:Request, file:UploadFile = File(...)):
    image = read_image(file)
    landmarks = ModelSingleton()(image)

    landmark_array = [landmark[:3,:] for landmark in landmarks]
    results =[]
    for face in landmark_array:
        aligned_cat_face = align_face(image, face)
        uid = str(uuid.uuid4())
        face_store[uid] = aligned_cat_face
        url = request.url_for("get_face", face_id=uid)
        results.append(str(url))

    return {"images": results}

@router.get("/align_face/{face_id}", name="get_face")
def get_aligned_face(face_id):
    aligned_cat_face = face_store[face_id]

    return StreamingResponse(image_to_buffer(aligned_cat_face), media_type="image/png")