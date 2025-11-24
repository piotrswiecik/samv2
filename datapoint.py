"""
This module contains a simple script to visualize data points in ARCADE training dataset.
The dataset yields a random image and a random class selection (from annotations available for this image).
Then a random prompt point is selected from all available points in binary mask.
"""

from typing import Annotated
import numpy as np
import typer
from matplotlib import pyplot as plt
import random

from dataset import ArcadeDataset, get_dataset_paths


def main(dataset_root: Annotated[str, typer.Option(prompt="Path to dataset root")]):
    ann_path, img_path = get_dataset_paths(dataset_root)
    dataset = ArcadeDataset(ann_path, img_path)

    sample = random.randint(0, len(dataset) - 1)
    sample = dataset[sample]

    # while not interrupted, show random samples
    while True:
        sample = random.randint(0, len(dataset) - 1)
        sample = dataset[sample]

        img_vis = sample["image"].permute(1, 2, 0).numpy().astype(np.uint8)
        mask_vis = sample["mask"].squeeze().numpy()  # [1024, 1024]
        point = sample["point"][0]

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_vis)
        plt.scatter([point[0]], [point[1]], c="r", s=50, label="Prompt")
        plt.title("Resized Image (1024x1024)")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_vis, cmap="gray")
        plt.scatter([point[0]], [point[1]], c="r", s=50)
        plt.title("Resized Mask (1024x1024)")
        plt.show()


if __name__ == "__main__":
    typer.run(main)
