from __future__ import annotations

import argparse
import csv
import hashlib
import math
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data_loading import load_task_examples_jsonl
from .run_decoupled_eval import config_hash, finalize_mode_summary
from .stats import paired_bootstrap_delta, paired_mcnemar
from .utils import ensure_dir, load_yaml, resolve_path


sns.set_theme(style="whitegrid", context="talk")

DEFAULT_CONFIG_PATHS = [
    "configs/confirmatory_v3_qwen7b.yaml",
    "configs/confirmatory_v3_qwen14b.yaml",
    "configs/confirmatory_v3_deepseek7b.yaml",
    "configs/confirmatory_v3_ministral8b.yaml",
]
CORE_CLAIM_CONDITIONS = [
    "no_cue",
    "exact_repetition",
    "generic_neutral",
    "matched_semantic",
    "single_canonical",
    "matched_procedural",
    "matched_lexical_overlap",
]
SECONDARY_CONTROL_CONDITIONS = [
    "matched_mixed",
    "matched_near_miss",
]
PRIMARY_MODES = ["free_form_only", "cot_before_options"]
STAT_CONTROLS = [
    "no_cue",
    "single_canonical",
    "matched_procedural",
    "matched_lexical_overlap",
    "exact_repetition",
]
CONTRAST_MAP = {
    "matched_semantic_vs_no_cue": "semantic_minus_nocue",
    "matched_semantic_vs_single_canonical": "canonical_minus_semantic",
    "matched_semantic_vs_matched_procedural": "procedural_minus_semantic",
    "matched_semantic_vs_matched_lexical_overlap": "lexical_minus_semantic",
}
USECOLS = [
    "model_alias",
    "model_id",
    "eval_seed",
    "config_hash",
    "task_name",
    "family_name",
    "item_id",
    "difficulty",
    "condition",
    "cue_family",
    "cue_type",
    "evaluation_mode",
    "cue_placement",
    "formatting_style",
    "final_correct",
    "solve_correct",
    "binding_correct",
    "parse_failure_solve",
    "parse_failure_bind",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        action="append",
        default=[],
        help="Expected confirmatory_v3 config paths. If omitted, the active four-model registry is used.",
    )
    parser.add_argument(
        "--combined-summary-csv",
        default="results/processed/confirmatory_v3/combined_family_summary.csv",
    )
    parser.add_argument(
        "--run-manifest-csv",
        default="results/processed/confirmatory_v3/run_manifest.csv",
    )
    parser.add_argument(
        "--integrity-check-csv",
        default="results/processed/confirmatory_v3/integrity_check.csv",
    )
    parser.add_argument(
        "--model-registry-csv",
        default="results/processed/confirmatory_v3/model_registry.csv",
    )
    parser.add_argument(
        "--core-claim-table-csv",
        default="results/processed/confirmatory_v3/core_claim_table.csv",
    )
    parser.add_argument(
        "--secondary-control-table-csv",
        default="results/processed/confirmatory_v3/secondary_control_table.csv",
    )
    parser.add_argument(
        "--paired-stats-primary-csv",
        default="results/processed/confirmatory_v3/paired_stats_primary.csv",
    )
    parser.add_argument(
        "--paired-stats-all-modes-csv",
        default="results/processed/confirmatory_v3/paired_stats_all_modes.csv",
    )
    parser.add_argument(
        "--meta-summary-csv",
        default="results/processed/confirmatory_v3/cross_model_meta_summary.csv",
    )
    parser.add_argument(
        "--competence-csv",
        default="results/processed/confirmatory_v3/competence_gating.csv",
    )
    parser.add_argument(
        "--primary-verdict-csv",
        default="results/processed/confirmatory_v3/model_primary_verdict_table.csv",
    )
    parser.add_argument(
        "--figures-dir",
        default="figures/confirmatory_v3",
    )
    parser.add_argument(
        "--report-path",
        default="reports/phaseR1_multimodel_confirmatory.md",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=5000,
    )
    return parser.parse_args()


def verdict_from_ci(ci_low: float, ci_high: float, delta: float) -> str:
    if ci_low > 0:
        return "effect_exists"
    if ci_high < 0:
        return "likely_null_or_reversed"
    if abs(delta) < 1e-12:
        return "uncertain"
    return "uncertain"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected_registry(config_paths: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path_text in config_paths:
        config_path = resolve_path(path_text)
        config = load_yaml(config_path)
        input_path = resolve_path(config["data"]["synthetic_path"])
        input_examples = load_task_examples_jsonl(input_path)
        prompt_template_path = resolve_path(config["prompt"]["templates_path"])
        cue_bundle_path = resolve_path(config["prompt"]["bundles_v2_path"])
        rows.append(
            {
                "model_alias": config["model"]["alias"],
                "model_id": config["model"]["model_id"],
                "config_path": config_path.relative_to(resolve_path(".")).as_posix(),
                "seed": int(config["experiment"]["seed"]),
                "expected_config_hash": config_hash(config),
                "prompt_template_path": prompt_template_path.relative_to(resolve_path(".")).as_posix(),
                "prompt_template_hash": sha256_file(prompt_template_path),
                "cue_bundle_path": cue_bundle_path.relative_to(resolve_path(".")).as_posix(),
                "cue_bundle_hash": sha256_file(cue_bundle_path),
                "input_jsonl_path": input_path.relative_to(resolve_path(".")).as_posix(),
                "input_jsonl_hash": sha256_file(input_path),
                "input_item_count": int(len(input_examples)),
                "conditions": "|".join(config["evaluation"]["conditions"]),
                "evaluation_modes": "|".join(config["evaluation"]["evaluation_modes"]),
                "expected_condition_count": int(len(config["evaluation"]["conditions"])),
                "expected_mode_count": int(len(config["evaluation"]["evaluation_modes"])),
                "expected_row_count": int(len(input_examples) * len(config["evaluation"]["conditions"]) * len(config["evaluation"]["evaluation_modes"])),
                "expected_cue_placement": config["evaluation"]["cue_placement"],
                "expected_formatting_style": config["evaluation"]["formatting_style"],
                "raw_jsonl_path": config["results"]["raw_jsonl"],
                "items_csv_path": config["results"]["items_csv"],
                "summary_csv_path": config["results"]["summary_csv"],
                "family_summary_csv_path": config["results"]["family_summary_csv"],
                "report_path": config["results"]["report_path"],
                "command": (
                    "/workspace/.venv-gpu/bin/python -m src.run_decoupled_eval "
                    f"--config {config_path.relative_to(resolve_path('.')).as_posix()} "
                    "--gpu-id <gpu>"
                ),
                "model_registry_note": (
                    "open fallback replacing inaccessible gated models"
                    if config["model"]["alias"] == "ministral-8b-instruct-2410"
                    else "active registry model"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("model_alias").reset_index(drop=True)


def load_item_subset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=lambda column: column in USECOLS)


def build_run_manifest(registry: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    for record in registry.to_dict(orient="records"):
        items_path = resolve_path(record["items_csv_path"])
        raw_jsonl_path = resolve_path(record["raw_jsonl_path"])
        status = "missing"
        row = dict(record)
        row.update(
            {
                "items_csv_exists": items_path.exists(),
                "raw_jsonl_exists": raw_jsonl_path.exists(),
                "row_count": 0,
                "recorded_model_alias": "",
                "recorded_model_id": "",
                "recorded_seed": math.nan,
                "recorded_config_hash": "",
                "config_hash_match": False,
                "cue_placement_match": False,
                "formatting_style_match": False,
                "condition_coverage_complete": False,
                "mode_coverage_complete": False,
                "run_status": status,
                "last_modified_utc": "",
            }
        )
        if items_path.exists():
            frame = load_item_subset(items_path)
            frames[record["model_alias"]] = frame
            row["row_count"] = int(len(frame))
            row["recorded_model_alias"] = "|".join(sorted(frame["model_alias"].dropna().unique()))
            row["recorded_model_id"] = "|".join(sorted(frame["model_id"].dropna().unique()))
            seeds = sorted(int(value) for value in frame["eval_seed"].dropna().unique())
            config_hashes = sorted(str(value) for value in frame["config_hash"].dropna().unique())
            placements = sorted(str(value) for value in frame["cue_placement"].dropna().unique())
            formats = sorted(str(value) for value in frame["formatting_style"].dropna().unique())
            conditions = sorted(str(value) for value in frame["condition"].dropna().unique())
            modes = sorted(str(value) for value in frame["evaluation_mode"].dropna().unique())
            row["recorded_seed"] = seeds[0] if len(seeds) == 1 else "|".join(str(value) for value in seeds)
            row["recorded_config_hash"] = "|".join(config_hashes)
            row["config_hash_match"] = config_hashes == [record["expected_config_hash"]]
            row["cue_placement_match"] = placements == [record["expected_cue_placement"]]
            row["formatting_style_match"] = formats == [record["expected_formatting_style"]]
            row["condition_coverage_complete"] = set(conditions) == set(record["conditions"].split("|"))
            row["mode_coverage_complete"] = set(modes) == set(record["evaluation_modes"].split("|"))
            row["last_modified_utc"] = datetime.fromtimestamp(items_path.stat().st_mtime, tz=UTC).isoformat()
            if (
                row["row_count"] == record["expected_row_count"]
                and row["config_hash_match"]
                and row["cue_placement_match"]
                and row["formatting_style_match"]
                and row["condition_coverage_complete"]
                and row["mode_coverage_complete"]
            ):
                row["run_status"] = "completed"
            else:
                row["run_status"] = "partial_or_mismatched"
        manifest_rows.append(row)
    manifest = pd.DataFrame(manifest_rows).sort_values("model_alias").reset_index(drop=True)
    return manifest, frames


def expected_keys_for_model(record: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    examples = load_task_examples_jsonl(resolve_path(record["input_jsonl_path"]))
    keys = set()
    for example in examples:
        family_name = example.metadata.get("family_v2", example.concept_name) if example.metadata else example.concept_name
        for evaluation_mode in record["evaluation_modes"].split("|"):
            for condition in record["conditions"].split("|"):
                keys.add((family_name, evaluation_mode, condition, example.item_id))
    return keys


def build_integrity_check(registry: pd.DataFrame, manifest: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    expected_interface_values = set()
    for record in registry.to_dict(orient="records"):
        model_alias = record["model_alias"]
        frame = frames.get(model_alias)
        if frame is None:
            rows.append(
                {
                    "model_alias": model_alias,
                    "row_count": 0,
                    "expected_row_count": record["expected_row_count"],
                    "duplicate_key_count": math.nan,
                    "missing_key_count": math.nan,
                    "extra_key_count": math.nan,
                    "condition_coverage_complete": False,
                    "mode_coverage_complete": False,
                    "cue_placement_match": False,
                    "formatting_style_match": False,
                    "integrity_ok": False,
                }
            )
            continue
        key_cols = ["family_name", "evaluation_mode", "condition", "item_id"]
        actual_keys = set(map(tuple, frame[key_cols].to_records(index=False)))
        expected_keys = expected_keys_for_model(record)
        duplicate_key_count = int(len(frame) - frame[key_cols].drop_duplicates().shape[0])
        missing_key_count = int(len(expected_keys - actual_keys))
        extra_key_count = int(len(actual_keys - expected_keys))
        interface_pair = (
            "|".join(sorted(frame["cue_placement"].dropna().unique())),
            "|".join(sorted(frame["formatting_style"].dropna().unique())),
        )
        expected_interface_values.add(interface_pair)
        manifest_row = manifest[manifest["model_alias"] == model_alias].iloc[0]
        integrity_ok = (
            manifest_row["run_status"] == "completed"
            and duplicate_key_count == 0
            and missing_key_count == 0
            and extra_key_count == 0
        )
        rows.append(
            {
                "model_alias": model_alias,
                "row_count": int(len(frame)),
                "expected_row_count": int(record["expected_row_count"]),
                "duplicate_key_count": duplicate_key_count,
                "missing_key_count": missing_key_count,
                "extra_key_count": extra_key_count,
                "condition_coverage_complete": bool(manifest_row["condition_coverage_complete"]),
                "mode_coverage_complete": bool(manifest_row["mode_coverage_complete"]),
                "cue_placement_match": bool(manifest_row["cue_placement_match"]),
                "formatting_style_match": bool(manifest_row["formatting_style_match"]),
                "integrity_ok": bool(integrity_ok),
            }
        )
    rows.append(
        {
            "model_alias": "__cross_model_interface__",
            "row_count": math.nan,
            "expected_row_count": math.nan,
            "duplicate_key_count": math.nan,
            "missing_key_count": math.nan,
            "extra_key_count": math.nan,
            "condition_coverage_complete": True,
            "mode_coverage_complete": True,
            "cue_placement_match": len(expected_interface_values) == 1,
            "formatting_style_match": len(expected_interface_values) == 1,
            "integrity_ok": len(expected_interface_values) == 1,
        }
    )
    return pd.DataFrame(rows)


def load_completed_items(manifest: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    completed_aliases = manifest.loc[manifest["run_status"] == "completed", "model_alias"].tolist()
    if not completed_aliases:
        raise FileNotFoundError("No completed confirmatory_v3 item CSVs are ready for postprocessing.")
    return pd.concat([frames[alias] for alias in completed_aliases], ignore_index=True)


def build_core_claim_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep = summary[summary["condition"].isin(CORE_CLAIM_CONDITIONS)].copy()
    rows: list[dict[str, Any]] = []
    for (model_alias, family_name, evaluation_mode), group in keep.groupby(
        ["model_alias", "family_name", "evaluation_mode"],
        dropna=False,
    ):
        acc = {row["condition"]: row["final_accuracy"] for _, row in group.iterrows()}
        matched_semantic = acc.get("matched_semantic", float("nan"))
        single_canonical = acc.get("single_canonical", float("nan"))
        matched_procedural = acc.get("matched_procedural", float("nan"))
        matched_lexical = acc.get("matched_lexical_overlap", float("nan"))
        no_cue = acc.get("no_cue", float("nan"))
        exact = acc.get("exact_repetition", float("nan"))
        generic = acc.get("generic_neutral", float("nan"))
        best_condition = max(acc, key=acc.get)
        rows.append(
            {
                "model_alias": model_alias,
                "family_name": family_name,
                "evaluation_mode": evaluation_mode,
                "best_condition": best_condition,
                "best_accuracy": acc[best_condition],
                "matched_semantic": matched_semantic,
                "single_canonical": single_canonical,
                "matched_procedural": matched_procedural,
                "matched_lexical_overlap": matched_lexical,
                "no_cue": no_cue,
                "exact_repetition": exact,
                "generic_neutral": generic,
                "semantic_beats_main_controls": (
                    matched_semantic > no_cue
                    and matched_semantic > exact
                    and matched_semantic > generic
                    and matched_semantic > matched_procedural
                    and matched_semantic > matched_lexical
                    and matched_semantic > single_canonical
                ),
                "semantic_minus_nocue": matched_semantic - no_cue,
                "procedural_minus_semantic": matched_procedural - matched_semantic,
                "canonical_minus_semantic": single_canonical - matched_semantic,
                "lexical_minus_semantic": matched_lexical - matched_semantic,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["model_alias", "evaluation_mode", "family_name"]
    ).reset_index(drop=True)


def build_secondary_control_table(summary: pd.DataFrame) -> pd.DataFrame:
    keep = summary[summary["condition"].isin(["matched_semantic", *SECONDARY_CONTROL_CONDITIONS, "no_cue"])].copy()
    rows: list[dict[str, Any]] = []
    for (model_alias, family_name, evaluation_mode), group in keep.groupby(
        ["model_alias", "family_name", "evaluation_mode"],
        dropna=False,
    ):
        acc = {row["condition"]: row["final_accuracy"] for _, row in group.iterrows()}
        semantic = acc.get("matched_semantic", float("nan"))
        no_cue = acc.get("no_cue", float("nan"))
        for condition in SECONDARY_CONTROL_CONDITIONS:
            control = acc.get(condition, float("nan"))
            rows.append(
                {
                    "model_alias": model_alias,
                    "family_name": family_name,
                    "evaluation_mode": evaluation_mode,
                    "condition": condition,
                    "accuracy": control,
                    "semantic_accuracy": semantic,
                    "no_cue_accuracy": no_cue,
                    "secondary_minus_semantic": control - semantic,
                    "secondary_minus_nocue": control - no_cue,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model_alias", "evaluation_mode", "family_name", "condition"]
    ).reset_index(drop=True)


def paired_stats_for_slice(
    items: pd.DataFrame,
    label: str,
    bootstrap_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [
        ("family_mode", family_name, evaluation_mode, items[(items["family_name"] == family_name) & (items["evaluation_mode"] == evaluation_mode)].copy())
        for family_name in sorted(items["family_name"].dropna().unique())
        for evaluation_mode in sorted(items["evaluation_mode"].dropna().unique())
    ]
    scopes.append(("all_modes_aggregate", "all_families", "all_modes", items.copy()))
    scopes.append(("primary_aggregate", "all_families", "primary_modes", items[items["evaluation_mode"].isin(PRIMARY_MODES)].copy()))

    for scope, family_name, evaluation_mode, subset in scopes:
        if subset.empty:
            continue
        semantic = subset[subset["condition"] == "matched_semantic"][
            ["item_id", "evaluation_mode", "final_correct"]
        ].rename(columns={"final_correct": "semantic_correct"})
        for control in STAT_CONTROLS:
            control_frame = subset[subset["condition"] == control][
                ["item_id", "evaluation_mode", "final_correct"]
            ].rename(columns={"final_correct": "control_correct"})
            merge_cols = ["item_id"] if scope == "family_mode" else ["item_id", "evaluation_mode"]
            merged = semantic.merge(control_frame, on=merge_cols, how="inner")
            if merged.empty:
                continue
            semantic_values = merged["semantic_correct"].astype(int).to_numpy()
            control_values = merged["control_correct"].astype(int).to_numpy()
            bootstrap = paired_bootstrap_delta(
                semantic_values,
                control_values,
                n_boot=bootstrap_samples,
                seed=17,
            )
            delta = float(semantic_values.mean() - control_values.mean())
            rows.append(
                {
                    "model_alias": label,
                    "scope": scope,
                    "family_name": family_name,
                    "evaluation_mode": evaluation_mode,
                    "contrast": f"matched_semantic_vs_{control}",
                    "n_items": int(len(merged)),
                    "semantic_accuracy": float(semantic_values.mean()),
                    "control_accuracy": float(control_values.mean()),
                    "delta_semantic_minus_control": delta,
                    "bootstrap_delta_mean": bootstrap.delta_mean,
                    "bootstrap_ci_low": bootstrap.ci_low,
                    "bootstrap_ci_high": bootstrap.ci_high,
                    "mcnemar_pvalue": paired_mcnemar(semantic_values, control_values),
                    "statistical_verdict": verdict_from_ci(bootstrap.ci_low, bootstrap.ci_high, delta),
                }
            )
    return pd.DataFrame(rows)


def build_paired_stats(items: pd.DataFrame, bootstrap_samples: int) -> pd.DataFrame:
    frames = []
    for model_alias in sorted(items["model_alias"].dropna().unique()):
        model_items = items[items["model_alias"] == model_alias].copy()
        frames.append(paired_stats_for_slice(model_items, model_alias, bootstrap_samples))
    frames.append(paired_stats_for_slice(items, "pooled_all_models", bootstrap_samples))
    return pd.concat(frames, ignore_index=True).sort_values(
        ["model_alias", "scope", "family_name", "evaluation_mode", "contrast"]
    ).reset_index(drop=True)


def build_primary_verdict_table(stats_frame: pd.DataFrame) -> pd.DataFrame:
    primary = stats_frame[stats_frame["scope"] == "primary_aggregate"].copy()
    return primary.sort_values(["model_alias", "contrast"]).reset_index(drop=True)


def build_meta_summary(core_claim_table: pd.DataFrame, stats_frame: pd.DataFrame) -> pd.DataFrame:
    primary_claims = core_claim_table[core_claim_table["evaluation_mode"].isin(PRIMARY_MODES)].copy()
    primary_stats = stats_frame[stats_frame["scope"] == "primary_aggregate"].copy()
    per_model = primary_stats[primary_stats["model_alias"] != "pooled_all_models"].copy()
    pooled = primary_stats[primary_stats["model_alias"] == "pooled_all_models"].copy()

    rows: list[dict[str, Any]] = [
        {
            "contrast": "semantic_vs_all_main_controls",
            "models": int(primary_claims["model_alias"].nunique()),
            "mean_delta": float(primary_claims["semantic_minus_nocue"].mean()),
            "median_delta": float(primary_claims["semantic_minus_nocue"].median()),
            "effect_exists_count": int(primary_claims["semantic_beats_main_controls"].sum()),
            "likely_null_or_reversed_count": int(len(primary_claims) - primary_claims["semantic_beats_main_controls"].sum()),
            "uncertain_count": 0,
            "pooled_delta": float("nan"),
            "pooled_ci_low": float("nan"),
            "pooled_ci_high": float("nan"),
            "pooled_mcnemar_pvalue": float("nan"),
            "pooled_verdict": "descriptive_only",
        }
    ]

    for contrast, group in per_model.groupby("contrast", dropna=False):
        pooled_row = pooled[pooled["contrast"] == contrast]
        pooled_record = pooled_row.iloc[0] if not pooled_row.empty else None
        rows.append(
            {
                "contrast": contrast,
                "models": int(group["model_alias"].nunique()),
                "mean_delta": float(group["delta_semantic_minus_control"].mean()),
                "median_delta": float(group["delta_semantic_minus_control"].median()),
                "effect_exists_count": int((group["statistical_verdict"] == "effect_exists").sum()),
                "likely_null_or_reversed_count": int((group["statistical_verdict"] == "likely_null_or_reversed").sum()),
                "uncertain_count": int((group["statistical_verdict"] == "uncertain").sum()),
                "pooled_delta": float(pooled_record["delta_semantic_minus_control"]) if pooled_record is not None else float("nan"),
                "pooled_ci_low": float(pooled_record["bootstrap_ci_low"]) if pooled_record is not None else float("nan"),
                "pooled_ci_high": float(pooled_record["bootstrap_ci_high"]) if pooled_record is not None else float("nan"),
                "pooled_mcnemar_pvalue": float(pooled_record["mcnemar_pvalue"]) if pooled_record is not None else float("nan"),
                "pooled_verdict": str(pooled_record["statistical_verdict"]) if pooled_record is not None else "",
            }
        )
    return pd.DataFrame(rows).sort_values("contrast").reset_index(drop=True)


def competence_gating_frame(core_claim_table: pd.DataFrame) -> pd.DataFrame:
    primary = core_claim_table[core_claim_table["evaluation_mode"].isin(PRIMARY_MODES)].copy()
    rows: list[dict[str, Any]] = []
    for _, row in primary.iterrows():
        for cue_name, accuracy in [
            ("matched_semantic", row["matched_semantic"]),
            ("single_canonical", row["single_canonical"]),
            ("matched_procedural", row["matched_procedural"]),
            ("matched_lexical_overlap", row["matched_lexical_overlap"]),
        ]:
            rows.append(
                {
                    "model_alias": row["model_alias"],
                    "family_name": row["family_name"],
                    "evaluation_mode": row["evaluation_mode"],
                    "cue_name": cue_name,
                    "baseline_no_cue_accuracy": row["no_cue"],
                    "cue_delta": float(accuracy - row["no_cue"]),
                }
            )
    return pd.DataFrame(rows)


def save_competence_plot(frame: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=frame,
        x="baseline_no_cue_accuracy",
        y="cue_delta",
        hue="cue_name",
        style="model_alias",
        s=120,
    )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("No-Cue Accuracy")
    ax.set_ylabel("Cue Delta vs No-Cue")
    ax.set_title("Competence Gating Across Models and Primary Modes")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_binding_decomposition(summary: pd.DataFrame, output_path: Path) -> None:
    keep = summary[
        summary["condition"].isin(CORE_CLAIM_CONDITIONS)
        & summary["evaluation_mode"].isin(["binding_only", "cot_before_options"])
    ].copy()
    keep["metric_name"] = keep["evaluation_mode"].map(
        {
            "binding_only": "binding_only_final_accuracy",
            "cot_before_options": "cot_binding_given_correct_solve",
        }
    )
    keep["metric_value"] = keep.apply(
        lambda row: row["final_accuracy"]
        if row["evaluation_mode"] == "binding_only"
        else row["binding_accuracy_given_correct_solve"],
        axis=1,
    )
    aggregated = (
        keep.groupby(["model_alias", "condition", "metric_name"], dropna=False)["metric_value"]
        .mean()
        .reset_index()
    )
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(
        data=aggregated,
        x="model_alias",
        y="metric_value",
        hue="condition",
    )
    ax.set_title("Cross-Model Binding Decomposition")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=20)
    plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_primary_delta_heatmap(core_claim_table: pd.DataFrame, output_path: Path) -> None:
    primary = core_claim_table[core_claim_table["evaluation_mode"].isin(PRIMARY_MODES)].copy()
    aggregated = (
        primary.groupby(["model_alias", "family_name"], dropna=False)[
            ["semantic_minus_nocue", "canonical_minus_semantic", "procedural_minus_semantic", "lexical_minus_semantic"]
        ]
        .mean()
        .reset_index()
    )
    contrasts = [
        ("semantic_minus_nocue", "Semantic - No Cue"),
        ("canonical_minus_semantic", "Canonical - Semantic"),
        ("procedural_minus_semantic", "Procedural - Semantic"),
        ("lexical_minus_semantic", "Lexical - Semantic"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (column, title) in zip(axes.flatten(), contrasts, strict=True):
        pivot = aggregated.pivot(index="model_alias", columns="family_name", values=column)
        sns.heatmap(pivot, annot=True, fmt=".3f", center=0.0, cmap="coolwarm", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Family")
        ax.set_ylabel("Model")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_model_forest_by_contrast(stats_frame: pd.DataFrame, output_path: Path) -> None:
    primary = stats_frame[
        (stats_frame["scope"] == "primary_aggregate")
        & (stats_frame["model_alias"] != "pooled_all_models")
    ].copy()
    contrasts = [
        "matched_semantic_vs_no_cue",
        "matched_semantic_vs_single_canonical",
        "matched_semantic_vs_matched_procedural",
        "matched_semantic_vs_matched_lexical_overlap",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, contrast in zip(axes.flatten(), contrasts, strict=True):
        subset = primary[primary["contrast"] == contrast].sort_values("delta_semantic_minus_control")
        ax.errorbar(
            subset["delta_semantic_minus_control"],
            subset["model_alias"],
            xerr=[
                subset["delta_semantic_minus_control"] - subset["bootstrap_ci_low"],
                subset["bootstrap_ci_high"] - subset["delta_semantic_minus_control"],
            ],
            fmt="o",
            color="tab:blue",
            ecolor="tab:gray",
            capsize=3,
        )
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(contrast)
        ax.set_xlabel("Delta")
        ax.set_ylabel("Model")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def save_family_forest_by_contrast(stats_frame: pd.DataFrame, output_path: Path) -> None:
    family = stats_frame[
        (stats_frame["scope"] == "family_mode")
        & (stats_frame["evaluation_mode"].isin(PRIMARY_MODES))
    ].copy()
    contrasts = [
        "matched_semantic_vs_no_cue",
        "matched_semantic_vs_single_canonical",
        "matched_semantic_vs_matched_procedural",
        "matched_semantic_vs_matched_lexical_overlap",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, contrast in zip(axes.flatten(), contrasts, strict=True):
        subset = family[family["contrast"] == contrast].copy()
        subset["label"] = subset["model_alias"] + " | " + subset["family_name"] + " | " + subset["evaluation_mode"]
        subset = subset.sort_values("delta_semantic_minus_control")
        ax.errorbar(
            subset["delta_semantic_minus_control"],
            subset["label"],
            xerr=[
                subset["delta_semantic_minus_control"] - subset["bootstrap_ci_low"],
                subset["bootstrap_ci_high"] - subset["delta_semantic_minus_control"],
            ],
            fmt="o",
            color="tab:green",
            ecolor="tab:gray",
            capsize=2,
        )
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(contrast)
        ax.set_xlabel("Delta")
        ax.set_ylabel("Model | Family | Mode")
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=220)
    plt.close()


def build_report(
    report_path: Path,
    registry: pd.DataFrame,
    manifest: pd.DataFrame,
    integrity: pd.DataFrame,
    core_claim_table: pd.DataFrame,
    primary_stats: pd.DataFrame,
    meta_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    primary_claims = core_claim_table[core_claim_table["evaluation_mode"].isin(PRIMARY_MODES)].copy()
    model_summary = (
        primary_claims.groupby("model_alias", dropna=False)
        .agg(
            family_mode_cells=("model_alias", "count"),
            semantic_gt_all_count=("semantic_beats_main_controls", "sum"),
            canonical_gt_semantic_count=("canonical_minus_semantic", lambda series: int((series > 0).sum())),
            procedural_gt_semantic_count=("procedural_minus_semantic", lambda series: int((series > 0).sum())),
            lexical_gt_semantic_count=("lexical_minus_semantic", lambda series: int((series > 0).sum())),
        )
        .reset_index()
    )
    primary_verdicts = primary_stats[
        (primary_stats["scope"] == "primary_aggregate")
        & (primary_stats["model_alias"] != "pooled_all_models")
    ].copy()
    manifest_view = manifest[
        [
            "model_alias",
            "model_id",
            "run_status",
            "row_count",
            "expected_row_count",
            "config_hash_match",
            "condition_coverage_complete",
            "mode_coverage_complete",
        ]
    ].copy()
    registry_view = registry[
        [
            "model_alias",
            "model_id",
            "config_path",
            "expected_config_hash",
            "prompt_template_hash",
            "cue_bundle_hash",
            "input_jsonl_hash",
            "model_registry_note",
        ]
    ].copy()

    lines = [
        "# Phase R1 Multimodel Confirmatory",
        "",
        "## Benchmark Freeze",
        "",
        "- Frozen benchmark: `data/processed/confirmatory_v3/phaseE_heldout_4family_test.jsonl`",
        "- Scope: four-family confirmatory slice from construct-validity v3 (`boolean_logic`, `ordering_constraints`, `state_tracking`, `temporal_reasoning`).",
        "- Frozen interface: `after_question` cue placement, `plain_sentence` formatting, identical decoupled modes.",
        "- Primary paper-facing modes: `free_form_only`, `cot_before_options`.",
        "",
        "## Run Integrity",
        "",
        "```text",
        manifest_view.to_string(index=False),
        "```",
        "",
        "```text",
        integrity.to_string(index=False),
        "```",
        "",
        "## Model Registry",
        "",
        "```text",
        registry_view.to_string(index=False),
        "```",
        "",
        "## Availability / Fallback",
        "",
        "- `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it` were inaccessible at weight-download time in this environment, so the active four-model registry uses `mistralai/Ministral-8B-Instruct-2410` as the open fallback.",
        "",
        "## Relation To Phase R0",
        "",
        "- Legacy `phaseE` numbers remain directionally useful but not publication-grade because two shard-level config hashes cannot be reconstructed from the current frozen configs.",
        "- The clean `confirmatory_v3_qwen7b` rerun is therefore the publication reference baseline for the Phase E-style benchmark.",
        "",
        "## Model-Level Primary Summary",
        "",
        "```text",
        model_summary.to_string(index=False),
        "```",
        "",
        "## Primary Aggregate Verdicts",
        "",
        "```text",
        primary_verdicts.to_string(index=False),
        "```",
        "",
        "## Cross-Model Meta Summary",
        "",
        "```text",
        meta_summary.to_string(index=False),
        "```",
        "",
        "## Figures",
        "",
        f"- [competence gating]({(figures_dir / 'competence_gating.png').as_posix()})",
        f"- [binding decomposition]({(figures_dir / 'binding_decomposition.png').as_posix()})",
        f"- [primary delta heatmap]({(figures_dir / 'primary_delta_heatmap.png').as_posix()})",
        f"- [model forest by contrast]({(figures_dir / 'model_forest_by_contrast.png').as_posix()})",
        f"- [family forest by contrast]({(figures_dir / 'family_forest_by_contrast.png').as_posix()})",
    ]
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_paths = args.config_path or DEFAULT_CONFIG_PATHS
    registry = load_expected_registry(config_paths)
    manifest, frames = build_run_manifest(registry)
    integrity = build_integrity_check(registry, manifest, frames)

    run_manifest_csv = resolve_path(args.run_manifest_csv)
    integrity_check_csv = resolve_path(args.integrity_check_csv)
    model_registry_csv = resolve_path(args.model_registry_csv)
    ensure_dir(run_manifest_csv.parent)
    manifest.to_csv(run_manifest_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(integrity_check_csv.parent)
    integrity.to_csv(integrity_check_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(model_registry_csv.parent)
    registry.to_csv(model_registry_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    if not (manifest["run_status"] == "completed").all():
        missing = manifest.loc[manifest["run_status"] != "completed", ["model_alias", "run_status"]]
        raise ValueError(
            "confirmatory_v3 postprocess requires all registry models to be completed. "
            f"Outstanding runs: {missing.to_dict(orient='records')}"
        )
    if not integrity["integrity_ok"].fillna(False).all():
        bad = integrity.loc[~integrity["integrity_ok"].fillna(False), "model_alias"].tolist()
        raise ValueError(
            "confirmatory_v3 integrity check failed; refusing to aggregate partial or mismatched runs. "
            f"Bad entries: {bad}"
        )

    items = load_completed_items(manifest, frames)
    summary = finalize_mode_summary(items)
    core_claim_table = build_core_claim_table(summary)
    secondary_control_table = build_secondary_control_table(summary)
    paired_stats_all_modes = build_paired_stats(items, bootstrap_samples=args.bootstrap_samples)
    paired_stats_primary = paired_stats_all_modes[
        (paired_stats_all_modes["scope"] == "primary_aggregate")
        | (
            (paired_stats_all_modes["scope"] == "family_mode")
            & (paired_stats_all_modes["evaluation_mode"].isin(PRIMARY_MODES))
        )
    ].copy()
    meta_summary = build_meta_summary(core_claim_table, paired_stats_all_modes)
    competence = competence_gating_frame(core_claim_table)
    primary_verdicts = build_primary_verdict_table(paired_stats_all_modes)

    combined_summary_csv = resolve_path(args.combined_summary_csv)
    core_claim_table_csv = resolve_path(args.core_claim_table_csv)
    secondary_control_table_csv = resolve_path(args.secondary_control_table_csv)
    paired_stats_primary_csv = resolve_path(args.paired_stats_primary_csv)
    paired_stats_all_modes_csv = resolve_path(args.paired_stats_all_modes_csv)
    meta_summary_csv = resolve_path(args.meta_summary_csv)
    competence_csv = resolve_path(args.competence_csv)
    primary_verdict_csv = resolve_path(args.primary_verdict_csv)
    report_path = resolve_path(args.report_path)
    figures_dir = ensure_dir(args.figures_dir)

    ensure_dir(combined_summary_csv.parent)
    summary.to_csv(combined_summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(core_claim_table_csv.parent)
    core_claim_table.to_csv(core_claim_table_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(secondary_control_table_csv.parent)
    secondary_control_table.to_csv(secondary_control_table_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(paired_stats_primary_csv.parent)
    paired_stats_primary.to_csv(paired_stats_primary_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(paired_stats_all_modes_csv.parent)
    paired_stats_all_modes.to_csv(paired_stats_all_modes_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(meta_summary_csv.parent)
    meta_summary.to_csv(meta_summary_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(competence_csv.parent)
    competence.to_csv(competence_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    ensure_dir(primary_verdict_csv.parent)
    primary_verdicts.to_csv(primary_verdict_csv, index=False, quoting=csv.QUOTE_MINIMAL)

    save_competence_plot(competence, figures_dir / "competence_gating.png")
    save_binding_decomposition(summary, figures_dir / "binding_decomposition.png")
    save_primary_delta_heatmap(core_claim_table, figures_dir / "primary_delta_heatmap.png")
    save_model_forest_by_contrast(paired_stats_all_modes, figures_dir / "model_forest_by_contrast.png")
    save_family_forest_by_contrast(paired_stats_all_modes, figures_dir / "family_forest_by_contrast.png")
    build_report(report_path, registry, manifest, integrity, core_claim_table, paired_stats_primary, meta_summary, figures_dir)


if __name__ == "__main__":
    main()
