"""Basic inference test - predict on a random image from ARCADE dataset using a fine-tuned model."""

from typing_extensions import Annotated
from hydra import initialize_config_dir
import sam2
import typer
import os
import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2
from collections import defaultdict
from hydra.core.global_hydra import GlobalHydra

from checkpoints import CheckpointSizes, get_checkpoint_config
from clean_mask import clean_mask
from dataset import get_dataset_paths


class ArcadeInference:
    def __init__(self, size: CheckpointSizes, weights_path: str, device="cuda"):
        self.device = device
        self.target_size = 1024
        self.size = size

        checkpoint_config = get_checkpoint_config(size)
        checkpoint_path = os.path.join(
            "workdir", "artifacts", "checkpoints", checkpoint_config.filename
        )

        print(f"Config: {checkpoint_config.config_file}")
        print(f"Checkpoint: {checkpoint_path}")
        # config_path = os.path.join(
        #     os.path.dirname(__file__), "sam2", "configs", checkpoint_config.config_file
        # )

        sam2_dir = os.path.dirname(sam2.__file__)
        config_dir = os.path.join(sam2_dir, "configs")
        
        GlobalHydra.instance().clear()
        initialize_config_dir(config_dir=config_dir, version_base="1.2")

        # print(f"Building model from: {config_path}")
        # sam2_dir = os.path.dirname(sam2.__file__)
        # config_dir = os.path.join(sam2_dir, "configs")
        # print(f"Config Directory found at: {config_dir}")

        # GlobalHydra.instance().clear()
        # initialize_config_dir(config_dir=config_dir, version_base="1.2")

        try:
            self.model = build_sam2(
                config_file=checkpoint_config.config_file,
                ckpt_path=checkpoint_path,
                device=device,
            )
            print("Base architecture loaded")
        except Exception as e:
            print(f"Error: {e}")
            raise e
        
        self.model.sam_mask_decoder.use_high_res_features = False
        print(f"Loading fine-tuned weights from: {weights_path}")

        state_dict = torch.load(weights_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("Model ready.")

    def predict(self, image_path, point_coords):
        """
        image_path: Path to .png file
        point_coords: tuple (x, y) in original image coordinates
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        img_resized = cv2.resize(image, (self.target_size, self.target_size))

        img_tensor = (
            torch.tensor(img_resized, dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        scale_x = self.target_size / orig_w
        scale_y = self.target_size / orig_h

        px, py = point_coords
        px_scaled = px * scale_x
        py_scaled = py * scale_y

        point_tensor = torch.tensor([[[px_scaled, py_scaled]]], dtype=torch.float32).to(
            self.device
        )
        label_tensor = torch.tensor([[1]], dtype=torch.int32).to(self.device)

        with torch.no_grad():
            # encoder
            backbone_out = self.model.image_encoder(img_tensor)

            if isinstance(backbone_out, dict):
                image_embed = backbone_out["vision_features"]
                # if self.size in [CheckpointSizes.TINY, CheckpointSizes.SMALL]:
                #     high_res_feats = None
                # else:
                    # high_res_feats = [x for x in backbone_out["backbone_fpn"][:2]]
            else:
                image_embed = backbone_out

            # prompt encoder
            sparse_emb, dense_emb = self.model.sam_prompt_encoder(
                points=(point_tensor, label_tensor),
                boxes=None,
                masks=None,
            )

            # mask decoder
            low_res_masks, iou_preds, _, _ = self.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
                repeat_image=False,
                high_res_features=None, 
            )

            # upscale
            mask_1024 = torch.nn.functional.interpolate(
                low_res_masks,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            )

            mask_np = mask_1024.squeeze().cpu().numpy()

            mask_final = cv2.resize(mask_np, (orig_w, orig_h))

            binary_mask = (mask_final > 0.0).astype(np.uint8)

            solid_mask = clean_mask(binary_mask)

            return image, binary_mask, solid_mask, (px, py)


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
