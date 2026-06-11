"""Visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF

from .model import PATCH_GRID_SIZE


def visualize_patch_edges(
    image_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    grid_size: int = PATCH_GRID_SIZE,
    save_path: str | Path | None = None,
) -> None:
    image = TF.to_pil_image(image_tensor.detach().cpu())
    width, height = image.size

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    step_x = width / grid_size
    step_y = height / grid_size
    coords = [
        (col * step_x + step_x / 2, row * step_y + step_y / 2)
        for row in range(grid_size)
        for col in range(grid_size)
    ]

    for edge_i, edge_j in edge_index.t():
        x1, y1 = coords[edge_i.item()]
        x2, y2 = coords[edge_j.item()]
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=1)

    ax.axis("off")
    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close(fig)
