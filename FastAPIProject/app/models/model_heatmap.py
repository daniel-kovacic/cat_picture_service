from typing import Any

import torch
import torchvision
from torch import nn
from torchvision.models import ResNet


class ResNetHeatmap(nn.Module):

    def __init__(self, output_shape: tuple, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        base_model = torchvision.models.resnet18()
        for param in list(base_model.children())[:-1]:
            param.requires_grad = False
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, output_shape[0], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.heatmap_head(x)
        return x
