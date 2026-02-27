from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class CatEmbedder(nn.Module):
    def __init__(self, emb_dim=256, pretrained=True):
        super().__init__()
        base = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, emb_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = F.normalize(x)
        return x


class ArcFace(nn.Module):
    def __init__(self, emb_dim, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)

        self.s = s
        self.m = m

    def forward(self, embeddings, labels):
        W = F.normalize(self.W)

        cosine = F.linear(embeddings, W)

        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cosine)

        target_logits = torch.cos(theta + self.m)

        one_hot = F.one_hot(labels, num_classes=W.size(0)).float()

        logits = cosine * (1 - one_hot) + target_logits * one_hot

        logits *= self.s

        return logits


class CombinedClassifier(nn.Module):
    def __init__(self, num_classes, emb_dim=256, s=64.0, m=0.5, pretrained=True):
        super().__init__()
        self.embedder = CatEmbedder(emb_dim, pretrained=pretrained)
        self.head = ArcFace(emb_dim, num_classes, s=s, m=m)

    def forward(self, x, labels=None):
        emb = self.embedder(x)
        if labels is not None:
            return self.head(emb, labels)
        return emb
