import torch.nn as nn

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


class ScreenClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # Freeze backbone except last 2 blocks
        for param in self.model.features[:-2].parameters():
            param.requires_grad = False

        for param in self.model.features[-2:].parameters():
            param.requires_grad = True

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        return self.model(x)

    def unfreeze(self):
        for param in self.model.features.parameters():
            param.requires_grad = True