"""This module defines checkpoint configurations for pretrained SAM2 image encoders."""

from dataclasses import dataclass
from enum import Enum


class CheckpointSizes(str, Enum):
    TINY = "tiny"
    SMALL = "small"


@dataclass(frozen=True)
class CheckpointConfig:
    filename: str
    url: str
    config_file: str


def get_checkpoint_config(size: CheckpointSizes) -> CheckpointConfig:
    CHECKPOINT_CONFIGS = {
        "tiny": CheckpointConfig(
            filename="sam2_hiera_tiny.pt",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
            config_file="sam2/sam2_hiera_t.yaml",
        )
    }
    if size.value not in CHECKPOINT_CONFIGS:
        raise NotImplementedError(f"Unsupported checkpoint size: {size}")
    return CHECKPOINT_CONFIGS[size.value]
