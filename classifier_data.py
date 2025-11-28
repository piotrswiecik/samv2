"""
This module creates augmented dataset for classifier training.
Class 0 represents background (non-vessel) crops - run generate_background_class() to create it.
Other classes represent vessel crops extracted from ARCADE dataset annotations as per Syntax.
"""

import os
from typing_extensions import Annotated
import cv2
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random

import typer

from dataset import get_dataset_paths

TRAIN_JSON = "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/annotations/train.json"
TRAIN_IMAGES_DIR = (
    "/Users/piotrswiecik/dev/ives/coronary/datasets/arcade/syntax/train/images"
)
OUTPUT_DIR = "workdir/arcade_classifier_data"


def generate_background_class(
    json_path, image_dir, output_dir, num_samples_per_image=5
):
    # Setup Class 0 Folder
    bg_dir = os.path.join(output_dir, "0")
    os.makedirs(bg_dir, exist_ok=True)

    print(f"Loading JSON from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    # Map Image ID -> List of Vessel Polygons
    img_vessel_masks = {}

    # Pre-calculate the "Occupied" zones for every image
    print("Mapping occupied vessel zones...")
    for ann in tqdm(data["annotations"]):
        img_id = ann["image_id"]
        if img_id not in img_vessel_masks:
            img_vessel_masks[img_id] = []
        img_vessel_masks[img_id].append(ann["segmentation"])

    # Map ID -> Filename
    img_map = {img["id"]: img for img in data["images"]}

    print(f"Generating Background (Class 0) samples...")
    count = 0

    keys = list(img_vessel_masks.keys())
    # Shuffle to get random sampling if you want to stop early
    random.shuffle(keys)

    for img_id in tqdm(keys):
        if img_id not in img_map:
            continue

        # 1. Load Image
        filename = img_map[img_id]["file_name"]
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        # 2. Create "Forbidden Zone" Mask (All vessels combined)
        forbidden_mask = np.zeros((h, w), dtype=np.uint8)

        polygons = img_vessel_masks[img_id]
        for seg in polygons:
            for s in seg:
                poly = np.array(s).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(forbidden_mask, [poly], color=255)

        # Dilate the forbidden zone slightly to avoid getting too close to vessels
        kernel = np.ones((15, 15), np.uint8)
        forbidden_mask = cv2.dilate(forbidden_mask, kernel, iterations=1)

        # 3. Random Sampling
        samples_found = 0
        attempts = 0

        while samples_found < num_samples_per_image and attempts < 50:
            attempts += 1

            # Pick random crop size (simulating vessel crop sizes)
            crop_size = random.randint(64, 200)

            # Pick random top-left corner
            if w - crop_size <= 0 or h - crop_size <= 0:
                break

            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)

            # Check intersection with Forbidden Zone
            crop_area_mask = forbidden_mask[y : y + crop_size, x : x + crop_size]

            # If crop contains ANY vessel pixels (255), reject it
            if np.any(crop_area_mask > 0):
                continue

            # 4. Accept & Save
            crop = image[y : y + crop_size, x : x + crop_size]

            # Save
            save_name = f"bg_{img_id}_{samples_found}.png"
            cv2.imwrite(os.path.join(bg_dir, save_name), crop)

            samples_found += 1
            count += 1

    print(f"Done. Generated {count} background samples in folder '0'.")


def generate_classifier_dataset(json_path, image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading JSON from {json_path}...")
    with open(json_path, "r") as f:
        data = json.load(f)

    img_to_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_to_anns[ann["image_id"]].append(ann)

    img_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    print(f"Processing {len(img_to_anns)} images to extract vessel crops...")

    count_saved = 0
    count_skipped = 0

    for img_id, anns in tqdm(img_to_anns.items()):
        if img_id not in img_id_to_filename:
            continue

        filename = img_id_to_filename[img_id]
        img_path = os.path.join(image_dir, filename)

        image = cv2.imread(img_path)
        if image is None:
            continue

        h, w = image.shape[:2]

        for ann in anns:
            cat_id = ann["category_id"]

            class_dir = os.path.join(output_dir, str(cat_id))
            os.makedirs(class_dir, exist_ok=True)

            mask = np.zeros((h, w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], color=1)

            masked_crop = np.zeros_like(image)
            masked_crop[mask == 1] = image[mask == 1]

            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0:
                count_skipped += 1
                continue

            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()

            pad = 10
            y_min = max(0, y_min - pad)
            y_max = min(h, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(w, x_max + pad)

            final_crop = masked_crop[y_min:y_max, x_min:x_max]

            if final_crop.shape[0] < 10 or final_crop.shape[1] < 10:
                count_skipped += 1
                continue

            save_name = f"{img_id}_{ann['id']}.png"
            cv2.imwrite(os.path.join(class_dir, save_name), final_crop)
            count_saved += 1

    print(f"\nDone. Saved {count_saved} crops. Skipped {count_skipped} (empty/tiny).")
    print(f"Data stored in: {output_dir}")


def main(
        dataset_root: Annotated[str, typer.Option(prompt="Path to dataset root")],
        out_dir: Annotated[str, typer.Option(prompt="Output directory for classifier data")],
    ):
    ann_path, img_path = get_dataset_paths(dataset_root)
    generate_background_class(ann_path, img_path, out_dir)


if __name__ == "__main__":
    typer.run(main)
    