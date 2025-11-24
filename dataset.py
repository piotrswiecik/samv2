import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


def get_dataset_paths(root: str) -> tuple[str, str]:
    """Based on ARCADE root path construct and validate paths to annotations file and training image dir."""
    if not os.path.isdir(root):
        raise ValueError(f"Root ARCADE directory not found: {root}")

    # Find annotation file by convention
    fpath = os.path.join(
        root, "syntax", "train", "annotations", "train.json"
    )
    if not os.path.isfile(fpath):
        raise ValueError(f"File not found: {fpath}")

    # Find train image dir by convention
    image_path = os.path.join(root, "syntax", "train", "images")
    if not os.path.isdir(image_path):
        raise ValueError(f"Trainingg image directory not found: {image_path}")
    
    return fpath, image_path



class ArcadeDataset(Dataset):
    def __init__(self, json_path, image_dir, target_size=1024):
        self.image_dir = image_dir
        self.target_size = target_size

        with open(json_path, "r") as f:
            data = json.load(f)

        self.img_info_map = {img["id"]: img for img in data["images"]}

        self.ann_map = defaultdict(list)
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id in self.img_info_map:
                self.ann_map[img_id].append(ann)

        self.valid_img_ids = sorted(list(self.ann_map.keys()))

        print(f"Index built. Found {len(self.valid_img_ids)} images with annotations.")

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]

        img_info = self.img_info_map[img_id]
        file_name = img_info["file_name"]

        img_path = os.path.join(self.image_dir, file_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]
        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h

        image_resized = cv2.resize(image, (self.target_size, self.target_size))

        anns = self.ann_map[img_id]

        mask = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        present_categories = list(set([ann["category_id"] for ann in anns]))
        if len(present_categories) > 0:
            target_category = np.random.choice(present_categories)

            for ann in anns:
                if ann["image_id"] != img_id:
                    continue

                if ann["category_id"] == target_category:
                    for seg in ann["segmentation"]:
                        poly = np.array(seg).reshape((-1, 2))

                        poly = poly.astype(np.float32)
                        poly[:, 0] *= scale_x
                        poly[:, 1] *= scale_y
                        poly = poly.astype(np.int32)

                        cv2.fillPoly(mask, [poly], color=1)

        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) > 0:
            random_idx = np.random.randint(0, len(x_indices))
            input_point = np.array([[x_indices[random_idx], y_indices[random_idx]]])
            input_label = np.array([1])
        else:
            input_point = np.array([[0, 0]])
            input_label = np.array([-1])

        return {
            "image": torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1),
            "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            "point": input_point,
            "label": input_label,
            "id": img_id,
        }
