from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import CatLandmarkDataset
from models.model import ResNetModel
from training.trainer import ResNetModelTrainer
from utils import create_valid_data_point_files, create_data_loader


class TestModelTrainer:
    @pytest.fixture
    def model_trainer(self, tmp_path: Path, model: ResNetModel):
        create_valid_data_point_files(tmp_path, 100)
        dataset_train = CatLandmarkDataset(tmp_path, split="train")
        dataset_val = CatLandmarkDataset(tmp_path, split="val")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        return ResNetModelTrainer(model, optimizer, dataset_train, dataset_val, criterion)

    def test_train_step(self, model_trainer: ResNetModelTrainer):
        train_loader = model_trainer.train_loader
        x, y = next(iter(train_loader))
        x = x.to(model_trainer.device, non_blocking=True)
        y = y.to(model_trainer.device, non_blocking=True)

        train_output = model_trainer.train_step(x, y)
        assert "loss" in train_output
