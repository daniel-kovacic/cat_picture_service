import torch
from torch import nn
from torch.utils.data import DataLoader


class ResNetModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor):
        if batch_x.dim() != 4:
            raise RuntimeError(f"Batch dimension should be 4")
        batch_shape = batch_x.shape
        if batch_shape[1] != 3:
            raise RuntimeError(f"first data dimension should be 3")

        self.optimizer.zero_grad()
        self.model.train()

        y_hat = self.model(batch_x)

        loss = self.criterion(y_hat, batch_y)
        loss.backward()

        self.optimizer.step()
        return {"loss": loss.item(),
                "prediction": y_hat.detach().numpy(),
                "target": batch_x.detach().numpy()}
