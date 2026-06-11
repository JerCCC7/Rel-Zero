"""Rel-Zero: relational zero-watermarking utilities."""

from .metrics import (
    binomial_tail_probability,
    compute_binomial_detection_threshold,
    compute_binomial_tpr,
    compute_edge_matching,
)
from .model import (
    IMAGE_SIZE,
    PATCH_GRID_SIZE,
    DEFAULT_TOP_K,
    RelZeroPipeline,
    build_pipeline,
    load_checkpoint,
    load_image,
    select_device,
)

__all__ = [
    "IMAGE_SIZE",
    "PATCH_GRID_SIZE",
    "DEFAULT_TOP_K",
    "RelZeroPipeline",
    "binomial_tail_probability",
    "build_pipeline",
    "compute_binomial_detection_threshold",
    "compute_binomial_tpr",
    "compute_edge_matching",
    "load_checkpoint",
    "load_image",
    "select_device",
]
