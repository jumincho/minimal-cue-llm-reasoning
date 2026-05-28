"""Concatenate the per-family benchmark JSONLs into one test set with a manifest.

Step (2) in the rerun flow:

    1. build_benchmark_v3 → per-family JSONLs under data/processed/construct_validity_v3/
    2. prepare_inputs     → one consolidated JSONL + a CSV manifest + a markdown report  <-- this file
    3. run_decoupled_eval

Emits:

- `data/processed/confirmatory_v3/phaseE_heldout_4family_test.jsonl`
  (one TaskExample per line, ready for the runner).
- `results/processed/confirmatory_v3/phaseE_heldout_4family_manifest.csv`
  (per-family count + sha256 of included item ids).
- `reports/confirmatory_v3_input_manifest.md`
  (human-readable summary of the manifest with the output sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .data_loading import load_task_examples_jsonl, save_task_examples_jsonl
from .utils import ensure_dir, resolve_path


DEFAULT_FAMILIES = [
    "boolean_logic",
    "ordering_constraints",
    "state_tracking",
    "temporal_reasoning",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/processed/construct_validity_v3",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        default=DEFAULT_FAMILIES,
    )
    parser.add_argument(
        "--output-jsonl",
        default="data/processed/confirmatory_v3/phaseE_heldout_4family_test.jsonl",
    )
    parser.add_argument(
        "--manifest-csv",
        default="results/processed/confirmatory_v3/phaseE_heldout_4family_manifest.csv",
    )
    parser.add_argument(
        "--report-path",
        default="reports/confirmatory_v3_input_manifest.md",
    )
    return parser.parse_args()


def hash_item_ids(item_ids: list[str]) -> str:
    payload = "\n".join(item_ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_jsonl = resolve_path(args.output_jsonl)
    manifest_csv = resolve_path(args.manifest_csv)
    report_path = resolve_path(args.report_path)

    examples = []
    manifest_rows: list[dict[str, object]] = []
    for family in args.families:
        family_path = input_dir / f"test_{family}.jsonl"
        family_examples = load_task_examples_jsonl(family_path)
        examples.extend(family_examples)
        difficulties = {}
        regimes = {}
        for example in family_examples:
            metadata = example.metadata or {}
            difficulties[metadata.get("difficulty", "unknown")] = difficulties.get(metadata.get("difficulty", "unknown"), 0) + 1
            regimes[metadata.get("lexicalization_regime", "unknown")] = regimes.get(metadata.get("lexicalization_regime", "unknown"), 0) + 1
        manifest_rows.append(
            {
                "family_name": family,
                "input_path": family_path.relative_to(resolve_path(".")).as_posix(),
                "n_items": len(family_examples),
                "item_id_hash": hash_item_ids([example.item_id for example in family_examples]),
                "difficulty_counts": json.dumps(difficulties, sort_keys=True),
                "lexicalization_counts": json.dumps(regimes, sort_keys=True),
            }
        )

    save_task_examples_jsonl(examples, output_jsonl)
    ensure_dir(manifest_csv.parent)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(manifest_csv, index=False)

    output_hash = hashlib.sha256(output_jsonl.read_bytes()).hexdigest()
    report_lines = [
        "# Confirmatory V3 Input Manifest",
        "",
        f"- Output JSONL: `{output_jsonl.relative_to(resolve_path('.')).as_posix()}`",
        f"- Families: {', '.join(args.families)}",
        f"- Total items: `{len(examples)}`",
        f"- Output sha256: `{output_hash}`",
        "",
        "## Family Manifest",
        "",
        "```text",
        manifest.to_string(index=False),
        "```",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
