#!/usr/bin/env python
"""Evaluate Rel-Zero on original/edited image pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from relzero.data import load_pair_manifest
from relzero.metrics import (
    compute_binomial_detection_threshold,
    compute_binomial_tpr,
    compute_edge_matching,
)
from relzero.model import (
    DEFAULT_TOP_K,
    PATCH_GRID_SIZE,
    build_pipeline,
    load_checkpoint,
    load_image,
    select_device,
)
from relzero.viz import visualize_patch_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Rel-Zero pair robustness.")
    parser.add_argument("--pairs", default="configs/example_pairs.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--grid-size", type=int, default=PATCH_GRID_SIZE)
    parser.add_argument("--negative-match-prob", type=float, default=0.06)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--vis-dir", default="outputs/pair_edges")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    pairs = load_pair_manifest(args.pairs)
    pipeline = build_pipeline(device)
    epoch = load_checkpoint(pipeline, args.checkpoint, device)

    threshold_matches, _ = compute_binomial_detection_threshold(
        num_bits=args.top_k,
        negative_match_prob=args.negative_match_prob,
        target_fpr=args.target_fpr,
    )

    print(f"device: {device}")
    print(f"checkpoint_epoch: {epoch}")
    print(f"pair_count: {len(pairs)}")
    print(
        f"binomial_threshold_matches: {threshold_matches}/{args.top_k} "
        f"target_fpr={args.target_fpr:.6f} "
        f"negative_match_prob={args.negative_match_prob:.4f}"
    )

    results: list[dict[str, object]] = []
    for pair in pairs:
        original = load_image(pair["original"], device)
        edited = load_image(pair["edited"], device)
        original_edges = pipeline.predict_edges(original, top_k=args.top_k)
        edited_edges = pipeline.predict_edges(edited, top_k=args.top_k)
        match = compute_edge_matching(edited_edges, original_edges)
        tpr_at_fpr = compute_binomial_tpr(
            match_prob=match["precision"],
            num_bits=args.top_k,
            min_matches=threshold_matches,
        )

        if args.save_vis:
            vis_dir = Path(args.vis_dir) / pair["name"]
            visualize_patch_edges(
                image_tensor=original,
                edge_index=original_edges,
                grid_size=args.grid_size,
                save_path=vis_dir / "original_edges.png",
            )
            visualize_patch_edges(
                image_tensor=edited,
                edge_index=edited_edges,
                grid_size=args.grid_size,
                save_path=vis_dir / "edited_edges.png",
            )

        result = {
            "name": pair["name"],
            "precision": match["precision"],
            "overlap": match["overlap"],
            "tpr_at_0.1_fpr": tpr_at_fpr,
        }
        results.append(result)
        print(
            f"[{pair['name']}] "
            f"precision={match['precision']:.4f} "
            f"overlap={int(match['overlap'])}/{args.top_k} "
            f"TPR@0.1%FPR={tpr_at_fpr:.4f}"
        )

    denom = max(len(results), 1)
    mean_precision = sum(float(item["precision"]) for item in results) / denom
    mean_tpr = sum(float(item["tpr_at_0.1_fpr"]) for item in results) / denom
    print(
        "mean "
        f"precision={mean_precision:.4f} "
        f"TPR@0.1%FPR={mean_tpr:.4f} "
        f"threshold_matches={threshold_matches}/{args.top_k}"
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
            "mean_precision": mean_precision,
            "mean_tpr_at_0.1_fpr": mean_tpr,
            "pairs": results,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"saved_json: {output_path}")


if __name__ == "__main__":
    main()
