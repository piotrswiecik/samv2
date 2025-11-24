import os
import sys
import typer

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from typing import Annotated


def main(dataset_root: Annotated[str, typer.Option(prompt="Path to dataset root")]):
    
    if not os.path.isdir(dataset_root):
        typer.echo(f"Directory not found: {dataset_root}")
        raise typer.Exit(1)

    # Find annotation file by convention
    ann_path = os.path.join(
        dataset_root, "syntax", "train", "annotations", "train.json"
    )
    if not os.path.isfile(ann_path):
        typer.echo(f"Annotations file not found, expected at {ann_path}")
        raise typer.Exit(1)

    # Find train image dir by convention
    image_path = os.path.join(dataset_root, "syntax", "train", "images")
    if not os.path.isdir(image_path):
        typer.echo(f"Image dir not found, expected at {image_path}")
        typer.Exit(1)

    typer.echo("OK, images & annotations found")


if __name__ == "__main__":
    typer.run(main)
