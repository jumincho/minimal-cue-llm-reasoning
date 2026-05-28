"""Lightweight I/O and reproducibility helpers shared across the pipeline.

Exposes:

- `PROJECT_ROOT`           : repo root, used to resolve config-relative paths.
- `resolve_path(p)`        : convert config-relative paths to absolute paths.
- `ensure_dir(p)`          : `mkdir -p` semantics.
- `load_yaml` / `dump_yaml`: small YAML wrappers.
- `read_jsonl` / `write_jsonl` / `append_jsonl`: line-delimited JSON helpers.
- `set_seed(seed)`         : seeds Python, NumPy, PyTorch, and CUDA deterministically.
- `normalize_text(s)`      : collapse whitespace.
- `stable_int_from_text(s)`: deterministic non-cryptographic hash, used for
                             item id seeding so a given item always lands in
                             the same shuffled position.
"""

from __future__ import annotations

import json
import random
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path_like: str | Path) -> Path:
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path_like: str | Path) -> dict[str, Any]:
    with resolve_path(path_like).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(data: dict[str, Any], path_like: str | Path) -> None:
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def read_jsonl(path_like: str | Path) -> list[dict[str, Any]]:
    path = resolve_path(path_like)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[dict[str, Any]], path_like: str | Path) -> None:
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path_like: str | Path, row: dict[str, Any]) -> None:
    path = resolve_path(path_like)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def stable_int_from_text(text: str, modulo: int | None = None) -> int:
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    if modulo is not None:
        return value % modulo
    return value
