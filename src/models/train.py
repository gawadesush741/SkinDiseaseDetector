import sys
sys.path.append("src")

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from models.skin_dataset import SkinDataset
from models.label_map import LABEL_MAP
from models.skin_model import create_model
from models.class_weights import calculate_class_weights
from preprocessing.transforms import image_transforms, train_transforms


# -----------------------------
# Settings
# -----------------------------

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.0001

IMAGE_FOLDER = "data/raw/images"


# -----------------------------
# Choose device
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Create datasets
# -----------------------------

train_dataset = SkinDataset(
    csv_file="data/raw/train.csv",
    image_folder=IMAGE_FOLDER,
    label_map=LABEL_MAP,
    transform=train_transforms
)

validation_dataset = SkinDataset(
    csv_file="data/raw/validation.csv",
    image_folder=IMAGE_FOLDER,
    label_map=LABEL_MAP,
    transform=image_transforms
)


# -----------------------------
# Create DataLoaders
# -----------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# Create model
# -----------------------------

model = create_model(
    num_classes=len(LABEL_MAP)
)

model = model.to(device)


# -----------------------------
# Class weights
# -----------------------------

class_weights = calculate_class_weights(
    "data/raw/train.csv",
    LABEL_MAP
)

class_weights = class_weights.to(device)


# -----------------------------
# Loss function
# -----------------------------

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# -----------------------------
# Optimizer
# -----------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training with Best Model Checkpointing
# -----------------------------

best_validation_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update model
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    average_loss = running_loss / len(train_loader)

    print("Training loss:", round(average_loss, 4))
    print("Training accuracy:", round(train_accuracy, 2), "%")


    # -----------------------------
    # Validation
    # -----------------------------

    model.eval()

    validation_correct = 0
    validation_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            validation_total += labels.size(0)
            validation_correct += (
                predicted == labels
            ).sum().item()

    validation_accuracy = (
        100 * validation_correct / validation_total
    )

    print(
        "Validation accuracy:",
        round(validation_accuracy, 2),
        "%"
    )


    # Checkpoint best model
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        torch.save(
            model.state_dict(),
            "skin_disease_model.pth"
        )
        print(f"--> Best model saved! (Validation Accuracy: {round(best_validation_accuracy, 2)}%)")


# -----------------------------
# Training Summary
# -----------------------------

print("\nTraining completed!")
print(f"Final Best Validation Accuracy: {round(best_validation_accuracy, 2)}%")
print("Best weights preserved in: skin_disease_model.pth")