"""This module contains training code for ResNet vessel classifier."""

import os
from typing import Annotated
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import typer


DATA_DIR = "/content/drive/MyDrive/ARCADE_Classifier_Data"  # The folder with 0, 1, 2...
CHECKPOINT_DIR = "/content/drive/MyDrive/ARCADE_SAM2_Project/checkpoints"
MODEL_SAVE_NAME = "vessel_classifier_resnet18.pth"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_classifier():
    print(f"🚀 Training Classifier on: {DEVICE}")

    # 1. Data Transforms
    # ResNet expects 224x224 inputs.
    # We add augmentation to make the model robust to rotation/flips.
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),  # Vessels can be slightly rotated
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # 2. Load Data
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms["train"])
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"📂 Found {len(full_dataset)} images across {num_classes} classes.")
    print(f"Classes: {class_names}")

    # Split 80/20 Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply 'val' transforms to validation set (Standard PyTorch trick)
    val_dataset.dataset.transform = data_transforms["val"]

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # 3. Setup Model (ResNet18)
    model = models.resnet18(weights="DEFAULT")

    # Modify the final layer to match our number of classes (26: 0-25)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(DEVICE)

    # 4. Optimizer & Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning Rate Scheduler (Decay LR by 0.1 every 7 epochs)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 5. Training Loop
    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in tqdm(dataloader, desc=phase):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + Optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Deep Copy Best Model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_path = os.path.join(CHECKPOINT_DIR, MODEL_SAVE_NAME)
                torch.save(model.state_dict(), save_path)
                print(f"💾 New Best Model Saved (Acc: {best_acc:.4f})")

    print(f"\n✅ Training Complete. Best Val Acc: {best_acc:.4f}")


def main(
    dataset_root: Annotated[str, typer.Option(prompt="Path to cropped dataset root")],
    learning_rate: Annotated[float, typer.Option(prompt="Learning rate")] = 1e-4,
    epochs: Annotated[int, typer.Option(prompt="Number of epochs")] = 10,
):
    train_classifier()


if __name__ == "__main__":
    typer.run(main)
