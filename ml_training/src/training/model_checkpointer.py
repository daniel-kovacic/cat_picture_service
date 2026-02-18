import datetime
import json
import os
from pathlib import Path

import torch
from torch import nn

from config import MODEL_REGISTRY_PATH


class ModelCheckpointer:
    def __init__(self, model_str: str, params: dict, metric="mse"):
        self.metric = metric
        self.run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model_dir = os.path.join(MODEL_REGISTRY_PATH, self.run_id)
        self.model_name = model_str
        self.best_metric = None
        self.params = params

    def _create_metadata_dict(self, metrics: dict[str, float], iteration: int):
        metadata_dict = {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "evaluation_metric": self.metric,
            "metrics": metrics,
            "iteration": iteration,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        }

        return metadata_dict

    def update(self, model: nn.Module, epoch: int, metrics: dict[str, float]):
        if self.best_metric is None or metrics[self.metric] < self.best_metric:
            if self.best_metric is not None:
                print(f"model with new best metric: {metrics[self.metric]} found at epoch {epoch}")
            self.best_metric = metrics[self.metric]
            metadata_dict = self._create_metadata_dict(metrics, epoch)
            self.save_model(model, metadata_dict)

    def save_model(self, model: nn.Module, metadata_dict: dict):
        run_file = Path(self.model_dir) / "run.json"
        run_file.parent.mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.model_dir, "run.json"), "w") as f:
            json.dump(metadata_dict, f)
        torch.save(model.state_dict(), os.path.join(self.model_dir, "model.pt"))

    def save_history(self, history: dict):
        path = os.path.join(self.model_dir, "history.json")
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
