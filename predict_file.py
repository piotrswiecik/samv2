"""Basic inference test - predict on a select image from file using a fine-tuned model."""

import cv2
from typing_extensions import Annotated
import typer
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from checkpoints import CheckpointSizes, get_checkpoint_config
from inference_model import ArcadeInference


def main(
    file: Annotated[str, typer.Option(prompt="Path to file")],
    size: Annotated[CheckpointSizes, typer.Option(prompt="Model size")],
    weights_path: str = typer.Option(..., prompt="Path to fine-tuned weights"),
):
    inference = ArcadeInference(
        size=size,
        weights_path=weights_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # open image, get size
    img = cv2.imread(file)
    h, w = img.shape[:2]

    # select random grid coordinate
    rand_h = np.random.randint(0, h)
    rand_w = np.random.randint(0, w)

    original_image, predicted_mask, solid_mask, click_point = inference.predict(
        file, (rand_w, rand_h)
    )

    plt.figure(figsize=(15, 7))

    # Original with Click
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.scatter(
        click_point[0],
        click_point[1],
        c="red",
        s=100,
        marker="x",
        label="User Click",
    )
    plt.legend()

    # Prediction Overlay
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    overlay = np.zeros_like(original_image)
    overlay[:, :, 1] = solid_mask * 255  # Green Overlay
    plt.imshow(overlay, alpha=0.6)
    plt.scatter(click_point[0], click_point[1], c="red", s=100, marker="x")
    plt.title("SAM2 Prediction")

    plt.show()


if __name__ == "__main__":
    typer.run(main)
