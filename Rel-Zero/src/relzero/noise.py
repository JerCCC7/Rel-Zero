"""Common distortion operators used in robustness tests."""

from __future__ import annotations

import io

import torch
from PIL import Image
import torchvision.transforms.functional as TF


def make_jpeg_tensor(image_tensor: torch.Tensor, quality: int) -> torch.Tensor:
    quality = max(1, min(int(quality), 100))
    image = TF.to_pil_image(image_tensor.detach().cpu())
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    jpeg_image = Image.open(buffer).convert("RGB")
    return TF.to_tensor(jpeg_image).to(image_tensor.device)


def make_gaussian_noise_tensor(image_tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    noise = torch.randn_like(image_tensor) * sigma
    return (image_tensor + noise).clamp(0.0, 1.0)
