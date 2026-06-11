#!/usr/bin/env python
"""Train the Rel-Zero pair predictor with VAE-generated stability labels."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CocoDetection

from relzero.metrics import compute_edge_matching
from relzero.model import (
    DEFAULT_TOP_K,
    IMAGE_SIZE,
    PATCH_GRID_SIZE,
    RelZeroPipeline,
    build_pipeline,
    freeze_module,
    generate_all_patch_pairs,
    select_device,
    set_eval_modes,
)


class IndexedCocoDataset(CocoDetection):
    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        image, _ = super().__getitem__(index)
        return index, image


class VAELabelModel(nn.Module):
    def __init__(
        self,
        model_name: str = "CompVis/stable-diffusion-v1-4",
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.deterministic = deterministic
        for param in self.vae.parameters():
            param.requires_grad_(False)
        self.vae.eval()

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        latent_dist = self.vae.encode(image).latent_dist
        latent = latent_dist.mean if self.deterministic else latent_dist.sample()
        return self.vae.decode(latent).sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Rel-Zero with VAE labels.")
    parser.add_argument("--coco-root", required=True)
    parser.add_argument("--coco-ann", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints/train_relzero")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)

    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--grid-size", type=int, default=PATCH_GRID_SIZE)
    parser.add_argument("--soft-temperature", type=float, default=0.2)
    parser.add_argument("--soft-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=0.05)
    parser.add_argument("--hard-neg-multiplier", type=int, default=4)
    parser.add_argument("--boundary-weight", type=float, default=0.1)
    parser.add_argument("--boundary-margin", type=float, default=0.02)
    parser.add_argument("--boundary-width", type=int, default=50)

    parser.add_argument("--lr-edge", type=float, default=1e-4)
    parser.add_argument("--lr-vit", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--freeze-student-vit", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")

    parser.add_argument("--vae-model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--vae-sampling", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def cosine_similarity(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    norm_a = features_a / (features_a.norm(dim=1, keepdim=True) + 1e-6)
    norm_b = features_b / (features_b.norm(dim=1, keepdim=True) + 1e-6)
    return (norm_a * norm_b).sum(dim=1)


def build_soft_target_distribution(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.softmax(scores / temperature, dim=0)


def build_vae_pair_target_item(
    feat_orig: torch.Tensor,
    feat_edit: torch.Tensor,
    edge_index: torch.Tensor,
    top_k: int,
    soft_temperature: float,
    boundary_width: int,
) -> dict[str, torch.Tensor]:
    edge_feat_orig = torch.abs(feat_orig[edge_index[0]] - feat_orig[edge_index[1]])
    edge_feat_edit = torch.abs(feat_edit[edge_index[0]] - feat_edit[edge_index[1]])
    similarity_scores = cosine_similarity(edge_feat_orig, edge_feat_edit)
    sorted_idx = torch.argsort(similarity_scores.detach(), descending=True)
    k = min(top_k, similarity_scores.numel())
    bw = min(boundary_width, k, similarity_scores.numel() - k)
    boundary_pos_start = max(0, k - bw)

    return {
        "soft_targets": build_soft_target_distribution(
            similarity_scores,
            soft_temperature,
        ).detach().cpu(),
        "topk_idx": sorted_idx[:k].detach().cpu(),
        "boundary_pos_idx": sorted_idx[boundary_pos_start:k].detach().cpu(),
        "boundary_neg_idx": sorted_idx[k : k + bw].detach().cpu(),
    }


@torch.no_grad()
def build_vae_label_target_cache(
    pipeline: RelZeroPipeline,
    vae_label_model: VAELabelModel,
    dataloader: DataLoader | None,
    device: torch.device,
    top_k: int,
    soft_temperature: float,
    boundary_width: int,
) -> dict[int, dict[str, torch.Tensor]]:
    cache: dict[int, dict[str, torch.Tensor]] = {}
    if dataloader is None:
        return cache

    set_eval_modes(pipeline)
    vae_label_model.eval()
    for sample_indices, orig_imgs in dataloader:
        orig_imgs = orig_imgs.to(device, non_blocking=True)
        vae_imgs = vae_label_model(orig_imgs)
        feat_orig = pipeline.teacher_vit(orig_imgs)
        feat_vae = pipeline.teacher_vit(vae_imgs)

        for sample_index, fo, fv in zip(sample_indices.tolist(), feat_orig, feat_vae):
            edge_index = generate_all_patch_pairs(fo.size(0), fo.device)
            cache[int(sample_index)] = build_vae_pair_target_item(
                feat_orig=fo,
                feat_edit=fv,
                edge_index=edge_index,
                top_k=top_k,
                soft_temperature=soft_temperature,
                boundary_width=boundary_width,
            )
    return cache


def build_train_val_dataloaders(
    coco_root: str | Path,
    coco_ann: str | Path,
    batch_size: int,
    num_workers: int,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader | None]:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    dataset = IndexedCocoDataset(root=str(coco_root), annFile=str(coco_ann), transform=transform)

    total_size = len(dataset)
    train_size = total_size if train_size <= 0 or train_size > total_size else train_size
    remaining = total_size - train_size
    val_size = max(0, min(val_size, remaining))
    discard_size = total_size - train_size - val_size

    split_generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, _ = random_split(
        dataset,
        [train_size, val_size, discard_size],
        generator=split_generator,
    )
    train_generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker,
        generator=train_generator,
    )
    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_worker,
        )
    return train_loader, val_loader


def compute_soft_listwise_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits / temperature, dim=0).unsqueeze(0)
    return F.kl_div(log_probs, target_probs.unsqueeze(0), reduction="batchmean")


def compute_sampled_hard_topk_bce_loss(
    logits: torch.Tensor,
    target_topk_idx: torch.Tensor,
    neg_multiplier: int,
) -> torch.Tensor:
    pos_idx = torch.unique(target_topk_idx.to(logits.device))
    pos_count = pos_idx.numel()
    if pos_count == 0 or neg_multiplier <= 0:
        return logits.new_tensor(0.0)

    neg_mask = torch.ones(logits.numel(), dtype=torch.bool, device=logits.device)
    neg_mask[pos_idx] = False
    neg_pool = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
    neg_count = min(neg_pool.numel(), pos_count * neg_multiplier)
    if neg_count == 0:
        return logits.new_tensor(0.0)

    neg_rank = torch.topk(logits[neg_pool].detach(), neg_count).indices
    neg_idx = neg_pool[neg_rank]
    selected_logits = torch.cat([logits[pos_idx], logits[neg_idx]], dim=0)
    labels = torch.cat(
        [
            torch.ones(pos_count, device=logits.device),
            torch.zeros(neg_count, device=logits.device),
        ],
        dim=0,
    )
    pos_weight = torch.tensor([neg_count / max(float(pos_count), 1.0)], device=logits.device)
    return F.binary_cross_entropy_with_logits(
        selected_logits,
        labels,
        pos_weight=pos_weight,
    )


def compute_boundary_ranking_loss(
    logits: torch.Tensor,
    pos_idx: torch.Tensor,
    neg_idx: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if pos_idx.numel() == 0 or neg_idx.numel() == 0:
        return logits.new_tensor(0.0)
    pos_scores = logits[pos_idx]
    neg_scores = logits[neg_idx]
    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        return logits.new_tensor(0.0)
    return F.relu(margin - pos_scores[:, None] + neg_scores[None, :]).mean()


def set_train_modes(pipeline: RelZeroPipeline, freeze_student_vit: bool) -> None:
    pipeline.teacher_vit.eval()
    pipeline.edge_model.train()
    if freeze_student_vit:
        pipeline.student_vit.eval()
    else:
        pipeline.student_vit.train()


def compute_batch_objective(
    pipeline: RelZeroPipeline,
    sample_indices: torch.Tensor,
    orig_imgs: torch.Tensor,
    target_cache: dict[int, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    feat_orig = pipeline.student_vit(orig_imgs)
    edge_index = generate_all_patch_pairs(feat_orig.size(1), feat_orig.device)
    sample_losses = []
    metric_sums = {
        "soft": 0.0,
        "hard": 0.0,
        "boundary": 0.0,
        "topk_precision": 0.0,
        "topk_iou": 0.0,
    }

    for sample_index, sample_features in zip(sample_indices.tolist(), feat_orig):
        cache_item = target_cache[int(sample_index)]
        target_probs = cache_item["soft_targets"].to(sample_features.device)
        target_topk_idx = cache_item["topk_idx"].to(sample_features.device)
        boundary_pos_idx = cache_item["boundary_pos_idx"].to(sample_features.device)
        boundary_neg_idx = cache_item["boundary_neg_idx"].to(sample_features.device)

        logits = pipeline.edge_model(sample_features, edge_index)
        soft_loss = compute_soft_listwise_loss(logits, target_probs, args.soft_temperature)
        hard_loss = compute_sampled_hard_topk_bce_loss(
            logits,
            target_topk_idx,
            args.hard_neg_multiplier,
        )
        boundary_loss = compute_boundary_ranking_loss(
            logits,
            boundary_pos_idx,
            boundary_neg_idx,
            args.boundary_margin,
        )
        total_loss = (
            args.soft_weight * soft_loss
            + args.hard_weight * hard_loss
            + args.boundary_weight * boundary_loss
        )
        sample_losses.append(total_loss)

        with torch.no_grad():
            pred_topk_idx = torch.topk(logits, min(args.top_k, logits.numel())).indices
            match = compute_edge_matching(
                edge_index[:, pred_topk_idx],
                edge_index[:, target_topk_idx],
            )
            union_count = (2 * args.top_k) - match["overlap"]
            metric_sums["soft"] += soft_loss.detach().item()
            metric_sums["hard"] += hard_loss.detach().item()
            metric_sums["boundary"] += boundary_loss.detach().item()
            metric_sums["topk_precision"] += float(match["precision"])
            metric_sums["topk_iou"] += float(match["overlap"]) / max(float(union_count), 1.0)

    loss = torch.stack(sample_losses).mean()
    sample_count = max(len(sample_losses), 1)
    metrics = {key: value / sample_count for key, value in metric_sums.items()}
    metrics["loss"] = loss.detach().item()
    return loss, metrics


def update_epoch_metrics(
    epoch_metrics: dict[str, float],
    batch_metrics: dict[str, float],
    batch_size: int,
) -> None:
    epoch_metrics["sample_count"] += batch_size
    for key, value in batch_metrics.items():
        epoch_metrics[key] = epoch_metrics.get(key, 0.0) + value * batch_size


def finalize_epoch_metrics(epoch_metrics: dict[str, float]) -> dict[str, float]:
    sample_count = max(epoch_metrics.pop("sample_count"), 1)
    return {key: value / sample_count for key, value in epoch_metrics.items()}


def train_one_epoch(
    pipeline: RelZeroPipeline,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_cache: dict[int, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, float]:
    set_train_modes(pipeline, freeze_student_vit=args.freeze_student_vit)
    epoch_metrics: dict[str, float] = {"sample_count": 0.0}
    for sample_indices, orig_imgs in dataloader:
        sample_indices = sample_indices.to(device)
        orig_imgs = orig_imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, batch_metrics = compute_batch_objective(
            pipeline=pipeline,
            sample_indices=sample_indices,
            orig_imgs=orig_imgs,
            target_cache=target_cache,
            args=args,
        )
        loss.backward()
        optimizer.step()
        update_epoch_metrics(epoch_metrics, batch_metrics, orig_imgs.size(0))
    return finalize_epoch_metrics(epoch_metrics)


@torch.no_grad()
def evaluate_one_epoch(
    pipeline: RelZeroPipeline,
    dataloader: DataLoader | None,
    device: torch.device,
    target_cache: dict[int, dict[str, torch.Tensor]],
    args: argparse.Namespace,
) -> dict[str, float] | None:
    if dataloader is None:
        return None
    set_eval_modes(pipeline)
    epoch_metrics: dict[str, float] = {"sample_count": 0.0}
    for sample_indices, orig_imgs in dataloader:
        sample_indices = sample_indices.to(device)
        orig_imgs = orig_imgs.to(device, non_blocking=True)
        _, batch_metrics = compute_batch_objective(
            pipeline=pipeline,
            sample_indices=sample_indices,
            orig_imgs=orig_imgs,
            target_cache=target_cache,
            args=args,
        )
        update_epoch_metrics(epoch_metrics, batch_metrics, orig_imgs.size(0))
    return finalize_epoch_metrics(epoch_metrics)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: str | Path,
    device: torch.device,
) -> int:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", 0))


def make_checkpoint_dir(root: str | Path, run_name: str | None) -> Path:
    root = Path(root)
    if run_name is None:
        run_name = f"relzero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_dir = root / run_name
    suffix = 1
    candidate = checkpoint_dir
    while candidate.exists():
        candidate = root / f"{run_name}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def format_metrics(prefix: str, metrics: dict[str, float] | None) -> str:
    if metrics is None:
        return ""
    return (
        f"{prefix}_loss={metrics['loss']:.4f} "
        f"{prefix}_soft={metrics['soft']:.4f} "
        f"{prefix}_hard={metrics['hard']:.4f} "
        f"{prefix}_boundary={metrics['boundary']:.4f} "
        f"{prefix}_precision={metrics['topk_precision']:.4f} "
        f"{prefix}_iou={metrics['topk_iou']:.4f}"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=args.deterministic)

    device = select_device(args.device)
    pipeline = build_pipeline(device, pretrained_backbone=not args.no_pretrained_backbone)
    if args.freeze_student_vit:
        freeze_module(pipeline.student_vit)

    vae_label_model = VAELabelModel(
        model_name=args.vae_model,
        deterministic=not args.vae_sampling,
    ).to(device)
    train_loader, val_loader = build_train_val_dataloaders(
        coco_root=args.coco_root,
        coco_ann=args.coco_ann,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    checkpoint_dir = make_checkpoint_dir(args.checkpoint_dir, args.run_name)
    param_groups = [
        {
            "params": pipeline.edge_model.parameters(),
            "lr": args.lr_edge,
            "weight_decay": args.weight_decay,
        }
    ]
    if not args.freeze_student_vit:
        param_groups.append(
            {
                "params": pipeline.student_vit.parameters(),
                "lr": args.lr_vit,
                "weight_decay": args.weight_decay,
            }
        )
    optimizer = torch.optim.AdamW(param_groups)

    start_epoch = 0
    if args.resume:
        start_epoch = load_training_checkpoint(pipeline, optimizer, args.resume, device)

    print(f"device: {device}")
    print(f"seed: {args.seed}")
    print(f"train_size: {len(train_loader.dataset)}")
    print(f"val_size: {0 if val_loader is None else len(val_loader.dataset)}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print("label_source: teacher_vit(original) vs teacher_vit(VAE(original))")
    print("predictor_input: original image only")
    print("building_vae_label_targets: start")
    train_cache = build_vae_label_target_cache(
        pipeline=pipeline,
        vae_label_model=vae_label_model,
        dataloader=train_loader,
        device=device,
        top_k=args.top_k,
        soft_temperature=args.soft_temperature,
        boundary_width=args.boundary_width,
    )
    val_cache = build_vae_label_target_cache(
        pipeline=pipeline,
        vae_label_model=vae_label_model,
        dataloader=val_loader,
        device=device,
        top_k=args.top_k,
        soft_temperature=args.soft_temperature,
        boundary_width=args.boundary_width,
    )
    print(f"building_vae_label_targets: done train_cache={len(train_cache)} val_cache={len(val_cache)}")

    best_val_precision = float("-inf")
    for epoch_idx in range(1, args.epochs + 1):
        epoch = start_epoch + epoch_idx
        train_metrics = train_one_epoch(
            pipeline=pipeline,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            target_cache=train_cache,
            args=args,
        )
        val_metrics = evaluate_one_epoch(
            pipeline=pipeline,
            dataloader=val_loader,
            device=device,
            target_cache=val_cache,
            args=args,
        )
        select_metrics = val_metrics if val_metrics is not None else train_metrics
        current_precision = select_metrics["topk_precision"]
        is_best = current_precision > best_val_precision
        if is_best:
            best_val_precision = current_precision

        message = [
            f"[RelZero {epoch_idx}/{args.epochs}]",
            format_metrics("train", train_metrics),
        ]
        if val_metrics is not None:
            message.append(format_metrics("val", val_metrics))
        message.append(f"best_val_precision={best_val_precision:.4f}")
        print(" ".join(message))

        save_checkpoint(pipeline, optimizer, epoch, checkpoint_dir / f"{epoch}.pth")
        if is_best:
            save_checkpoint(pipeline, optimizer, epoch, checkpoint_dir / "best_stage1.pth")
            print(f"Best checkpoint updated: epoch={epoch} val_topk_precision={current_precision:.4f}")


if __name__ == "__main__":
    main()
