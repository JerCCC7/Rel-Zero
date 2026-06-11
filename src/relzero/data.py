"""Input manifests for example pair evaluation."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_path(path: str | Path, base_dir: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_pair_manifest(manifest_path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    pairs = payload["pairs"] if isinstance(payload, dict) else payload
    base_dir = manifest_path.parent
    resolved: list[dict[str, str]] = []
    for item in pairs:
        resolved.append(
            {
                "name": item["name"],
                "original": str(resolve_path(item["original"], base_dir)),
                "edited": str(resolve_path(item["edited"], base_dir)),
            }
        )
    return resolved


def unique_images_from_pairs(pairs: list[dict[str, str]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    images: list[tuple[str, str]] = []
    for pair in pairs:
        for role in ("original", "edited"):
            image_path = pair[role]
            if image_path in seen:
                continue
            seen.add(image_path)
            images.append((f"{pair['name']}_{role}", image_path))
    return images
