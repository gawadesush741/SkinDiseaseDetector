import torch
import torch.nn as nn
from torchvision import models


def create_model(num_classes=7):

    model = models.resnet18(weights="DEFAULT")

    # Replace the final layer
    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        num_classes
    )

    return model