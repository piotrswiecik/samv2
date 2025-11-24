import os
import sys
import sam2
import typer
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from urllib.request import urlretrieve
from typing import Annotated

from checkpoints import CheckpointSizes, get_checkpoint_config
from dataset import ArcadeDataset, get_dataset_paths


def main(
    dataset_root: Annotated[str, typer.Option(prompt="Path to dataset root")],
    size: Annotated[CheckpointSizes, typer.Option(prompt="Model size")],
):
    try:
        ann_path, img_path = get_dataset_paths(dataset_root)
        dataset = ArcadeDataset(ann_path, img_path)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1)

    typer.echo("OK, dataset loaded")

    workdir = os.path.join(os.path.dirname(__file__), "workdir")
    artifact_dir = os.path.join(workdir, "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    checkpoint_dir = os.path.join(artifact_dir, "checkpoints")
    checkpoint_config = get_checkpoint_config(size)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_config.filename)

    if not os.path.isfile(checkpoint_file):
        typer.echo(f"Downloading checkpoint {checkpoint_config.url}...")
        urlretrieve(checkpoint_config.url, checkpoint_file)
        typer.echo(f"Checkpoint downloaded and cached to: {checkpoint_file}")
    else:
        typer.echo(f"Checkpoint already cached at: {checkpoint_file}")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    sam2_dir = os.path.dirname(sam2.__file__)
    config_dir = os.path.join(sam2_dir, "configs")
    print(f"Config Directory found at: {config_dir}")

    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=config_dir, version_base="1.2")

    try:
        sam2_model = build_sam2(
            config_file=checkpoint_config.config_file, 
            ckpt_path=checkpoint_file, 
            device=device
        )
        sam2_model.train()
        print("Model loaded successfully!")

        print("Running test pass...")
        dummy_images = torch.randn(1, 3, 1024, 1024).to(device)
        backbone_out = sam2_model.image_encoder(dummy_images)
        print("Forward pass complete. Pipeline is ready.")

    except Exception as e:
        print(f"Error: {e}")
        if os.path.exists(config_dir):
            print(f"Available configs: {os.listdir(config_dir)}")


if __name__ == "__main__":
    typer.run(main)
