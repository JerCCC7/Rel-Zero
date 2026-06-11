#!/usr/bin/env python
"""Evaluate Rel-Zero robustness under JPEG and Gaussian perturbations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from relzero.data import load_pair_manifest, unique_images_from_pairs
from relzero.metrics import (
    compute_binomial_detection_threshold,
    compute_binomial_tpr,
    compute_edge_matching,
)
from relzero.model import DEFAULT_TOP_K, build_pipeline, load_checkpoint, load_image, select_device
from relzero.noise import make_gaussian_noise_tensor, make_jpeg_tensor
from relzero.viz import visualize_patch_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Rel-Zero under simple distortions.")
    parser.add_argument("--pairs", default="configs/example_pairs.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--negative-match-prob", type=float, default=0.06)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--gaussian-sigma", type=float, default=0.05)
    parser.add_argument("--noise-seed", type=int, default=3407)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-dir", default="outputs/noise_edges")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def summarize(results: list[dict[str, object]], key: str) -> float:
    return sum(float(item[key]) for item in results) / max(len(results), 1)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    torch.manual_seed(args.noise_seed)

    pairs = load_pair_manifest(args.pairs)
    images = unique_images_from_pairs(pairs)
    pipeline = build_pipeline(device)
    epoch = load_checkpoint(pipeline, args.checkpoint, device)

    threshold_matches, _ = compute_binomial_detection_threshold(
        num_bits=args.top_k,
        negative_match_prob=args.negative_match_prob,
        target_fpr=args.target_fpr,
    )

    print(f"device: {device}")
    print(f"checkpoint_epoch: {epoch}")
    print(f"image_count: {len(images)}")
    print(f"jpeg_quality: {args.jpeg_quality}")
    print(f"gaussian_sigma: {args.gaussian_sigma}")
    print(
        f"binomial_threshold_matches: {threshold_matches}/{args.top_k} "
        f"target_fpr={args.target_fpr:.6f} "
        f"negative_match_prob={args.negative_match_prob:.4f}"
    )

    jpeg_results: list[dict[str, object]] = []
    gaussian_results: list[dict[str, object]] = []
    for image_name, image_path in images:
        image_tensor = load_image(image_path, device)
        original_edges = pipeline.predict_edges(image_tensor, top_k=args.top_k)

        jpeg_tensor = make_jpeg_tensor(image_tensor, quality=args.jpeg_quality)
        jpeg_edges = pipeline.predict_edges(jpeg_tensor, top_k=args.top_k)
        jpeg_match = compute_edge_matching(jpeg_edges, original_edges)
        jpeg_tpr = compute_binomial_tpr(
            match_prob=jpeg_match["precision"],
            num_bits=args.top_k,
            min_matches=threshold_matches,
        )

        gaussian_tensor = make_gaussian_noise_tensor(image_tensor, sigma=args.gaussian_sigma)
        gaussian_edges = pipeline.predict_edges(gaussian_tensor, top_k=args.top_k)
        gaussian_match = compute_edge_matching(gaussian_edges, original_edges)
        gaussian_tpr = compute_binomial_tpr(
            match_prob=gaussian_match["precision"],
            num_bits=args.top_k,
            min_matches=threshold_matches,
        )

        if args.save_vis:
            vis_dir = Path(args.vis_dir) / image_name
            visualize_patch_edges(image_tensor, original_edges, save_path=vis_dir / "original.png")
            visualize_patch_edges(jpeg_tensor, jpeg_edges, save_path=vis_dir / "jpeg.png")
            visualize_patch_edges(gaussian_tensor, gaussian_edges, save_path=vis_dir / "gaussian.png")

        jpeg_results.append(
            {
                "name": image_name,
                "precision": jpeg_match["precision"],
                "overlap": jpeg_match["overlap"],
                "tpr_at_0.1_fpr": jpeg_tpr,
            }
        )
        gaussian_results.append(
            {
                "name": image_name,
                "precision": gaussian_match["precision"],
                "overlap": gaussian_match["overlap"],
                "tpr_at_0.1_fpr": gaussian_tpr,
            }
        )
        print(
            f"[{image_name}] "
            f"jpeg_precision={jpeg_match['precision']:.4f} "
            f"jpeg_TPR@0.1%FPR={jpeg_tpr:.4f} "
            f"gaussian_precision={gaussian_match['precision']:.4f} "
            f"gaussian_TPR@0.1%FPR={gaussian_tpr:.4f}"
        )

    mean_jpeg_precision = summarize(jpeg_results, "precision")
    mean_jpeg_tpr = summarize(jpeg_results, "tpr_at_0.1_fpr")
    mean_gaussian_precision = summarize(gaussian_results, "precision")
    mean_gaussian_tpr = summarize(gaussian_results, "tpr_at_0.1_fpr")
    print(
        "mean "
        f"jpeg_precision={mean_jpeg_precision:.4f} "
        f"jpeg_TPR@0.1%FPR={mean_jpeg_tpr:.4f} "
        f"gaussian_precision={mean_gaussian_precision:.4f} "
        f"gaussian_TPR@0.1%FPR={mean_gaussian_tpr:.4f}"
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_epoch": epoch,
            "top_k": args.top_k,
            "negative_match_prob": args.negative_match_prob,
            "target_fpr": args.target_fpr,
            "threshold_matches": threshold_matches,
            "jpeg": {
                "quality": args.jpeg_quality,
                "mean_precision": mean_jpeg_precision,
                "mean_tpr_at_0.1_fpr": mean_jpeg_tpr,
                "images": jpeg_results,
            },
            "gaussian": {
                "sigma": args.gaussian_sigma,
                "mean_precision": mean_gaussian_precision,
                "mean_tpr_at_0.1_fpr": mean_gaussian_tpr,
                "images": gaussian_results,
            },
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_json: {output_path}")


if __name__ == "__main__":
    main()
