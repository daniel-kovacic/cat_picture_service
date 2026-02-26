import torch.nn
import tqdm
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam, Optimizer

from config import DATA_DIR, DATA_DIR_ID_CATS
from data.dataset import CatLandmarkDataset
from data.dataset_heatmap import CatLandmarkHeatmapDataset
from data.dataset_individual_cats import CatIdDataset
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
    optimizer = pick_optimizer(optimizer=optimizer_name, learning_rate=lr, model=model)
    if model_name == "resnet18":
        train_dataset = CatLandmarkDataset(DATA_DIR, "train")
        val_dataset = CatLandmarkDataset(DATA_DIR, "val")
        criterion = torch.nn.MSELoss()
        criterion_name = "mse"
        metrics = {criterion_name: MSELoss()}
        use_lbls_for_training = False
    elif model_name == "resnet18_heatmap":
        train_dataset = CatLandmarkHeatmapDataset(DATA_DIR, "train")
        val_dataset = CatLandmarkHeatmapDataset(DATA_DIR, "val")
        criterion = torch.nn.MSELoss()
        criterion_name = "mse"
        metrics = {criterion_name: MSELoss()}
        use_lbls_for_training = False
    elif model_name == "resnet18_arc_face":
        train_dataset = CatIdDataset(DATA_DIR_ID_CATS, "train")
        val_dataset = CatIdDataset(DATA_DIR_ID_CATS, "val")
        criterion = torch.nn.CrossEntropyLoss()
        criterion_name = "cross_ent"
        metrics = {criterion_name: torch.nn.CrossEntropyLoss()}

        use_lbls_for_training = True

    else:
        raise NotImplementedError()

    trainer = ResNetModelTrainer(model=model,
                                 optimizer=optimizer,
                                 train_dataset=train_dataset,
                                 val_dataset=val_dataset,
                                 criterion=criterion,
                                 batch_size=batch_size,
                                 metrics=metrics,
                                 use_lbls_for_training=use_lbls_for_training
                                 )
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    checkpointing = ModelCheckpointer(model_name, parameter_config["hyperparameters"], criterion_name)
    for epoch in tqdm.tqdm(range(epochs)):
        history["train_loss"].append(trainer.train_one_epoch())
        val_criterion = trainer.validate()[criterion_name]
        history["val_loss"].append(val_criterion)

        metrics = {criterion_name: val_criterion}
        checkpointing.update(model=model,
                             epoch=epoch,
                             metrics=metrics)
        print(f"train loss: {history['train_loss'][-1]}, val loss: {history['val_loss'][-1]}")
    checkpointing.save_history(history)


if __name__ == "__main__":
    main()
