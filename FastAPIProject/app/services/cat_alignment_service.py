from fastapi import APIRouter, UploadFile, File, FastAPI

from core.face_realignment import align_face


from core.util import read_image, fig_to_buf, image_to_buffer
from models.model_singleton import ModelSingleton
from fastapi.responses import StreamingResponse

router = APIRouter()

@router.post("/align_face")
def cat_annotation(file:UploadFile = File(...)):
    image = read_image(file)
    landmark = ModelSingleton()(image)
    landmark_array =landmark[:3,:].detach().cpu().numpy()
    aligned_cat_face = align_face(image, landmark_array)
    return StreamingResponse(image_to_buffer(aligned_cat_face), media_type="image/png")