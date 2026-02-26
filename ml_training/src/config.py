from pathlib import Path

LANDMARK_INDEX_DICT = {
    0: "left_eye",
    1: "right_eye",
    2: "mouth",
    3: "left_ear_1",
    4: "left_ear_2",
    5: "left_ear_3",
    6: "right_ear_1",
    7: "right_ear_2",
    8: "right_ear_3"
}

LANDMARK_COORD_SHAPE = (9, 2)

IMAGE_SHAPE = (224, 224)

CAT_ID_CLASSES = 564

MODEL_REGISTRY_PATH = Path(__file__).resolve().parent.parent.parent.joinpath("model_registry").resolve()
DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data").joinpath("raw").resolve()
DATA_DIR_ID_CATS = Path(__file__).resolve().parent.parent.joinpath("data").joinpath(
    "cat_individuals_dataset_preprocessed_v2").resolve()
LANDMARK_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.joinpath("models").joinpath(
    "landmark_model").resolve()
FACE_ID_MODEL_PATH = Path(__file__).resolve().parent.parent.parent.joinpath("models").joinpath(
    "face_id_model").resolve()
