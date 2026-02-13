from pathlib import Path

import numpy as np
import pytest
import torch

from models.model import ResNetModel
from training.trainer import ResNetModelTrainer
from utils import create_valid_data_point_files, create_data_loader


class TestModelTrainer:
    @pytest.fixture
    def model_trainer(self, tmp_path: Path, model: ResNetModel):
        create_valid_data_point_files(tmp_path, 100)
        data_loader_train = create_data_loader(tmp_path, split="train")
        data_loader_val = create_data_loader(tmp_path, split="val")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        return ResNetModelTrainer(model, data_loader_train, data_loader_val, criterion, optimizer)

    def test_train_step(self, model_trainer: ResNetModelTrainer):
        train_loader = model_trainer.train_loader
        model = model_trainer.model
        x, y = next(iter(train_loader))
        print("x_shape: ", x.shape)
        print("y_shape: ", y.shape)

        with torch.no_grad():
            y_hat = model(x)

        print(y_hat.shape)
        train_output = model_trainer.train_step(x, y)
        assert "loss" in train_output and "prediction" in train_output and "target" in train_output
        assert y_hat.shape == train_output["prediction"].shape
        assert np.allclose(train_output["prediction"], y_hat.numpy())
        with torch.no_grad():
            y_hat_trained = model(x)
        assert not np.allclose(y_hat, y_hat_trained.numpy())
