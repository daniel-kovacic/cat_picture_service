from typing import Any

import torch
import torchvision
from torch import nn
from torchvision.models import ResNet


class ResNetModel(nn.Module):

    def __init__(self, output_shape: tuple, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        base_model = torchvision.models.resnet18()
        for param in list(base_model.children())[:-1]:
            param.requires_grad = False
        num_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(nn.Linear(num_features, output_shape[0] * output_shape[1]),
                                      nn.Sigmoid())
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.model(x)
        x = x.view(batch_size, -1, 2)
        return x
