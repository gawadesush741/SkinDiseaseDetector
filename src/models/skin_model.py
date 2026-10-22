import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=7, pretrained=True):

    weights = "DEFAULT" if pretrained else None
    model = models.resnet18(weights=weights)

    # Replace the final layer
    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        num_classes
    )

    return model