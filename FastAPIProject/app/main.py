from fastapi import FastAPI

from core.config import Model_ID
from models.model_singleton import ModelSingleton
from services import annotation_service, face_cropping_service, cat_alignment_service

app = FastAPI()


from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = ModelSingleton()
    yield


app.include_router(annotation_service.router)
app.include_router(face_cropping_service.router)

app.include_router(cat_alignment_service.router)
