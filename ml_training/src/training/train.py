import json
from pathlib import Path
from typing import Literal

import torch
import tqdm
from fontTools.ufoLib import DATA_DIRNAME
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from config import LANDMARK_COORD_SHAPE, DATA_DIR
from data.dataset import CatLandmarkDataset
from models.model import ResNetModel
from training.model_checkpointer import ModelCheckpointer
from training.parameter_configs import parameter_configs
from training.trainer import ResNetModelTrainer


def pick_optimizer(optimizer: str, learning_rate: float, model: nn.Module) -> Optimizer:
    if optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Only adam optimizer is implemented.")
    return optimizer


def main():
    parameter_config = parameter_configs["config_1"]

    hyperparameters = parameter_config["hyperparameters"]

    model = parameter_config["model"]
    model_name = parameter_config["model_name"]
    lr = hyperparameters["lr"]
    batch_size = hyperparameters["batch_size"]
    optimizer_name = hyperparameters["optimizer"]
    epochs = hyperparameters["epochs"]
    criterion = parameter_config["criterion"]
    train_dataset = CatLandmarkDataset(DATA_DIR, "train")
    val_dataset = CatLandmarkDataset(DATA_DIR, "val")

    optimizer = pick_optimizer(optimizer=optimizer_name, learning_rate=lr, model=model)
    trainer = ResNetModelTrainer(model=model,
                                 optimizer=optimizer,
                                 train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 criterion=MSELoss(),
                                 batch_size=batch_size
                                 )
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    checkpointing = ModelCheckpointer(model_name, parameter_config["hyperparameters"], criterion)
    for epoch in tqdm.tqdm(range(epochs)):
        history["train_loss"].append(trainer.train_one_epoch())
        val_criterion = trainer.validate()[criterion]
        history["val_loss"].append(val_criterion)

        metrics = {criterion: val_criterion}
        checkpointing.update(model=model,
                             epoch=epoch,
                             metrics=metrics)
        print(f"train loss: {history['train_loss'][-1]}, val loss: {history['val_loss'][-1]}")
    checkpointing.save_history(history)


if __name__ == "__main__":
    main()
