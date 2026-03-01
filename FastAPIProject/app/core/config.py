from pathlib import Path

LANDMARK_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent.joinpath("models").joinpath(
    "landmark_model").resolve()
FACE_ID_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent.joinpath("models").joinpath(
    "face_id_model").resolve()
NUMBER_OF_CLASSES = 564

SIMILARITY_THRESHOLD = 0.270