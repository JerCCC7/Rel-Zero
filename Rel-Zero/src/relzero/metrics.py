"""Metrics for relational zero-watermark verification."""

from __future__ import annotations

import math

import torch


def compute_edge_matching(
    edges_a: torch.Tensor,
    edges_b: torch.Tensor,
) -> dict[str, float]:
    """Return precision-style overlap between two top-k edge sets."""
    if edges_a.size(1) == 0 or edges_b.size(1) == 0:
        return {"precision": 0.0, "overlap": 0.0}

    def normalize(edges: torch.Tensor) -> torch.Tensor:
        edges = edges.t().contiguous()
        return torch.stack(
            [torch.min(edges, dim=1)[0], torch.max(edges, dim=1)[0]],
            dim=1,
        )

    set_a = {tuple(pair) for pair in normalize(edges_a).tolist()}
    set_b = {tuple(pair) for pair in normalize(edges_b).tolist()}
    overlap = len(set_a & set_b)
    return {
        "precision": overlap / max(len(set_a), 1),
        "overlap": float(overlap),
    }


def binomial_tail_probability(
    num_bits: int,
    match_prob: float,
    min_matches: int,
) -> float:
    """Compute P[Binomial(num_bits, match_prob) >= min_matches]."""
    if num_bits <= 0:
        return 0.0

    p = min(max(match_prob, 0.0), 1.0)
    min_matches = max(0, min_matches)
    if min_matches <= 0:
        return 1.0
    if min_matches > num_bits:
        return 0.0

    q = 1.0 - p
    return sum(
        math.comb(num_bits, count)
        * (p**count)
        * (q ** (num_bits - count))
        for count in range(min_matches, num_bits + 1)
    )


def compute_binomial_detection_threshold(
    num_bits: int,
    negative_match_prob: float,
    target_fpr: float,
) -> tuple[int, float]:
    """Find the smallest match count whose null-tail probability is <= target_fpr."""
    for min_matches in range(0, num_bits + 2):
        actual_fpr = binomial_tail_probability(
            num_bits=num_bits,
            match_prob=negative_match_prob,
            min_matches=min_matches,
        )
        if actual_fpr <= target_fpr:
            return min_matches, actual_fpr
    return num_bits + 1, 0.0


def compute_binomial_tpr(
    match_prob: float,
    num_bits: int,
    min_matches: int,
) -> float:
    """Estimate TPR at a fixed FPR threshold from an observed matching probability."""
    return binomial_tail_probability(
        num_bits=num_bits,
        match_prob=match_prob,
        min_matches=min_matches,
    )
