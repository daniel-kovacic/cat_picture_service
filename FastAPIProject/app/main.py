from fastapi import FastAPI

from models.model_singleton import ModelSingleton
from services import annotation_service, face_cropping_service, face_alignment_service, similarity_service

app = FastAPI()


from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelSingleton()
    yield


app.include_router(annotation_service.router)
app.include_router(face_cropping_service.router)

app.include_router(face_alignment_service.router)

app.include_router(similarity_service.router)
