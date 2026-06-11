"""Model and image utilities for Rel-Zero inference."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 224
PATCH_GRID_SIZE = 14
DEFAULT_TOP_K = 50


class SingleImageEdgeMLP(nn.Module):
    """Pair scoring head used by the released Rel-Zero checkpoint."""

    def __init__(
        self,
        node_dim: int = 768,
        pair_hidden: int = 512,
        use_node_mlp: bool = True,
        node_hidden: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_node_mlp = use_node_mlp

        if use_node_mlp:
            self.node_mlp = nn.Sequential(
                nn.Linear(node_dim, node_hidden),
                nn.GELU(),
                nn.LayerNorm(node_hidden),
                nn.Dropout(dropout),
                nn.Linear(node_hidden, node_hidden),
                nn.GELU(),
                nn.LayerNorm(node_hidden),
            )
            feature_dim = node_hidden
        else:
            feature_dim = node_dim

        pair_in = feature_dim * 4
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, pair_hidden),
            nn.GELU(),
            nn.LayerNorm(pair_hidden),
            nn.Dropout(dropout),
            nn.Linear(pair_hidden, pair_hidden // 2),
            nn.GELU(),
            nn.Linear(pair_hidden // 2, 1),
        )

    @staticmethod
    def _pair_features(zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        return torch.cat([zi, zj, torch.abs(zi - zj), zi * zj], dim=-1)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_node_mlp:
            features = self.node_mlp(features)

        edge_i, edge_j = edge_index
        pair_features = self._pair_features(features[edge_i], features[edge_j])
        return self.pair_mlp(pair_features).squeeze(-1)


class ViTFeatureExtractor(nn.Module):
    """ViT patch-token feature extractor."""

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.vit.forward_features(x)
        return tokens[:, 1:]


def generate_all_patch_pairs(num_patches: int, device: torch.device) -> torch.Tensor:
    return torch.triu_indices(num_patches, num_patches, offset=1, device=device)


class RelZeroPipeline(nn.Module):
    """Predicts top-k stable patch pairs from a single image."""

    def __init__(
        self,
        teacher_vit: ViTFeatureExtractor,
        student_vit: ViTFeatureExtractor,
        edge_model: SingleImageEdgeMLP,
    ) -> None:
        super().__init__()
        self.teacher_vit = teacher_vit
        self.student_vit = student_vit
        self.edge_model = edge_model

    @torch.no_grad()
    def predict_edges(self, image: torch.Tensor, top_k: int) -> torch.Tensor:
        image_batch = image.unsqueeze(0)
        features = self.student_vit(image_batch).squeeze(0)
        edge_index = generate_all_patch_pairs(features.size(0), features.device)
        scores = self.edge_model(features, edge_index)
        top_k = min(top_k, scores.numel())
        topk_idx = torch.topk(scores, top_k).indices
        return edge_index[:, topk_idx]


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


def build_pipeline(device: torch.device, pretrained_backbone: bool = False) -> RelZeroPipeline:
    teacher_vit = ViTFeatureExtractor(pretrained=pretrained_backbone).to(device)
    student_vit = ViTFeatureExtractor(pretrained=pretrained_backbone).to(device)
    edge_model = SingleImageEdgeMLP(
        node_dim=768,
        pair_hidden=512,
        use_node_mlp=True,
        node_hidden=512,
        dropout=0.0,
    ).to(device)

    freeze_module(teacher_vit)
    return RelZeroPipeline(teacher_vit, student_vit, edge_model).to(device)


def set_eval_modes(pipeline: RelZeroPipeline) -> None:
    pipeline.eval()
    pipeline.teacher_vit.eval()
    pipeline.student_vit.eval()
    pipeline.edge_model.eval()


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> int:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    set_eval_modes(model)
    return int(checkpoint.get("epoch", 0))


def load_image(image_path: str | Path, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).to(device)


def select_device(requested_device: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
