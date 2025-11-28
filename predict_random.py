"""Basic inference test - predict on a random image from ARCADE dataset using a fine-tuned model."""

from typing_extensions import Annotated
import typer
import os
import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from checkpoints import CheckpointSizes
from dataset import get_dataset_paths
from inference_model import ArcadeInference


def main(
    dataset_root: Annotated[
        str, typer.Option(prompt="Path to dataset root")
    ],  # random pick from dataset
    size: Annotated[CheckpointSizes, typer.Option(prompt="Model size")],
    weights_path: str = typer.Option(..., prompt="Path to fine-tuned weights"),
):
    inference = ArcadeInference(
        size=size,
        weights_path=weights_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    ann_path, img_path = get_dataset_paths(dataset_root)

    with open(ann_path, "r") as f:
        data = json.load(f)

        img_map = {img["id"]: img for img in data["images"]}
        ann_map = defaultdict(list)
        for ann in data["annotations"]:
            ann_map[ann["image_id"]].append(ann)

        valid_ids = list(ann_map.keys())

        target_id = np.random.choice(valid_ids)
        img_info = img_map[target_id]
        anns = ann_map[target_id]

        target_ann = np.random.choice(anns)
        seg = target_ann["segmentation"][0]
        poly = np.array(seg).reshape((-1, 2))

        idx = np.random.randint(0, len(poly))
        click_x, click_y = poly[idx]

        full_path = os.path.join(img_path, img_info["file_name"])

        img_path, click_coords, img_id = full_path, (click_x, click_y), img_info["id"]

        original_image, predicted_mask, solid_mask, click_point = inference.predict(
            img_path, click_coords
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
        plt.title(f"Input Image (ID {img_id})")
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
