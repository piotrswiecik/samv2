"""This module defines checkpoint configurations for pretrained SAM2 image encoders."""

from dataclasses import dataclass
from enum import Enum


class CheckpointSizes(str, Enum):
    TINY = "tiny"
    SMALL = "small"
    BPLUS = "baseplus"
    LARGE = "large"


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
        ),
        "small": CheckpointConfig(
            filename="sam2_hiera_small.pt",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
            config_file="sam2.1/sam2.1_hiera_s.yaml",
        ),
        "baseplus": CheckpointConfig(
            filename="sam2_hiera_baseplus.pt",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
            config_file="sam2.1/sam2.1_hiera_b+.yaml",
        ),
        "large": CheckpointConfig(
            filename="sam2_hiera_large.pt",
            url="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
            config_file="sam2.1/sam2.1_hiera_l.yaml",
        ),
    }
    if size.value not in CHECKPOINT_CONFIGS:
        raise NotImplementedError(f"Unsupported checkpoint size: {size}")
    return CHECKPOINT_CONFIGS[size.value]
