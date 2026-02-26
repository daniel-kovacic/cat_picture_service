from collections import defaultdict
from typing import Callable, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training.metrics import avg_dist, mse

MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class ResNetModelTrainer:
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            train_dataset: Dataset,
            val_dataset: Dataset,
            criterion: nn.Module,
            batch_size: int = 32,
            shuffle: bool = True,
            num_workers: int = 0,
            use_lbls_for_training: bool = False,
            metrics: Dict[str, MetricFn] | None = None
    ) -> None:
        if torch.cuda.is_available():
            self.device: torch.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print("Using device:", self.device)

        self.model: nn.Module = model.to(self.device)
        self.optimizer: torch.optim.Optimizer = optimizer
        self.criterion: nn.Module = criterion

        self.use_lbls_for_training = use_lbls_for_training
        use_cuda: bool = self.device.type == "cuda"
        use_mps: bool = self.device.type == "mps"

        self.train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0 if use_mps else num_workers,
            pin_memory=use_cuda,
            persistent_workers=use_cuda and num_workers > 0,
            prefetch_factor=2 if use_cuda else None,
        )

        self.val_loader: DataLoader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0 if use_mps else num_workers,
            pin_memory=use_cuda
        )

        if metrics is None:
            metrics = {"avg_dist": avg_dist, "mse": mse}
        self.metrics: Dict[str, MetricFn] = metrics

    def train_step(
            self,
            batch_x: torch.Tensor,
            batch_y: torch.Tensor
    ) -> Dict[str, float]:
        batch_x = batch_x.to(self.device, non_blocking=True)
        batch_y = batch_y.to(self.device, non_blocking=True)

        self.model.train()
        self.optimizer.zero_grad()
        if not self.use_lbls_for_training:
            y_hat: torch.Tensor = self.model(batch_x)
        else:
            y_hat: torch.Tensor = self.model(batch_x, batch_y)

        loss: torch.Tensor = self.criterion(y_hat, batch_y)

        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item())}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_pts: int = 0

        metric_sums = {name: torch.zeros((), device=self.device)
                       for name in self.metrics}

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device, non_blocking=False)
                y = y.to(self.device, non_blocking=False)
                batch_size = x.shape[0]
                if not self.use_lbls_for_training:
                    y_hat: torch.Tensor = self.model(x)
                else:
                    y_hat: torch.Tensor = self.model(x, y)
                for name, metric in self.metrics.items():
                    res = metric(y_hat, y)
                    metric_sums[name] += res * batch_size
                total_pts += batch_size

        return {
            name: (metric_sums[name] / total_pts).item()
            for name in metric_sums
        }

    def train_one_epoch(self) -> float:
        total_loss: float = 0.0
        total_pts: int = 0

        for x, y in tqdm(list(self.train_loader), leave=False, desc="Training"):
            out = self.train_step(x, y)
            n: int = x.shape[0]
            total_loss += out["loss"] * n
            total_pts += n

        return total_loss / total_pts
