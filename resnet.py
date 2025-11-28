import os
from typing import Annotated
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import typer

MODEL_SAVE_NAME = "vessel_classifier_resnet18.pth"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_classifier(data_dir: str, resnet_checkpoint_dir: str):
    print(f"Training Classifier on: {DEVICE}")
    os.makedirs(resnet_checkpoint_dir, exist_ok=True)

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
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

    try:
        full_dataset = datasets.ImageFolder(
            data_dir, transform=data_transforms["train"]
        )
    except FileNotFoundError:
        print(f"Data directory not found at {data_dir}")
        return

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"Found {len(full_dataset)} images.")
    print(f"Detected {num_classes} classes: {class_names}")

    if "0" not in class_names:
        print("WARNING: Class '0' (Background) is missing!")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    class ValDatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform

        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y

        def __len__(self):
            return len(self.subset)

    raw_val_set = datasets.ImageFolder(data_dir)
    _, raw_val_subset = random_split(raw_val_set, [train_size, val_size])
    val_dataset = ValDatasetWrapper(raw_val_subset, transform=data_transforms["val"])

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = models.resnet18(weights="DEFAULT")

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=phase, leave=False):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "train":
                history["train_loss"].append(epoch_loss)
            if phase == "val":
                history["val_acc"].append(epoch_acc.item())

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                save_path = os.path.join(resnet_checkpoint_dir, MODEL_SAVE_NAME)
                torch.save(model.state_dict(), save_path)
                print(f"Saved New Best Model (Acc: {best_acc:.4f})")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.4f}")

    # Plot History
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"])
    plt.title("Train Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"])
    plt.title("Val Accuracy")
    plt.show()


def main(
    dataset_root: Annotated[str, typer.Option(prompt="Path to dataset root")],
    resnet_checkpoint_dir: Annotated[
        str, typer.Option(prompt="Path to save ResNet checkpoint")
    ],
):
    train_classifier(dataset_root, resnet_checkpoint_dir)


if __name__ == "__main__":
    typer.run(main)
