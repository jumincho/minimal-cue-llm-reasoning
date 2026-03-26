from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


@dataclass
class BootstrapResult:
    delta_mean: float
    ci_low: float
    ci_high: float


def aggregate_accuracy(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)
    summary = grouped["is_correct"].agg(["mean", "sum", "count"]).reset_index()
    return summary.rename(columns={"mean": "accuracy", "sum": "num_correct", "count": "n"})


def compute_margin(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby(["model_alias", "task_name", "condition"], dropna=False)
    return grouped["confidence_margin"].agg(["mean", "std", "count"]).reset_index()


def selective_steering_index(summary: pd.DataFrame) -> pd.DataFrame:
    concept_rows = summary[summary["condition_type"] == "concept_bundle"].copy()
    off_diag = (
        concept_rows[concept_rows["is_matched"] == 0]
        .groupby(["model_alias", "task_name"], dropna=False)["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "off_diagonal_mean"})
    )
    matched = concept_rows[concept_rows["is_matched"] == 1][
        ["model_alias", "task_name", "accuracy"]
    ].rename(columns={"accuracy": "matched_accuracy"})
    merged = matched.merge(off_diag, on=["model_alias", "task_name"], how="left")
    merged["ssi"] = merged["matched_accuracy"] - merged["off_diagonal_mean"]
    return merged


def paired_bootstrap_delta(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 7,
) -> BootstrapResult:
    rng = np.random.default_rng(seed)
    deltas = []
    indices = np.arange(len(a))
    for _ in range(n_boot):
        sample = rng.choice(indices, size=len(indices), replace=True)
        deltas.append(float(a[sample].mean() - b[sample].mean()))
    deltas_arr = np.asarray(deltas)
    return BootstrapResult(
        delta_mean=float(deltas_arr.mean()),
        ci_low=float(np.quantile(deltas_arr, 0.025)),
        ci_high=float(np.quantile(deltas_arr, 0.975)),
    )


def paired_mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> float:
    table = np.zeros((2, 2), dtype=int)
    for a_val, b_val in zip(correct_a.astype(int), correct_b.astype(int), strict=True):
        table[a_val, b_val] += 1
    result = mcnemar(table, exact=False, correction=True)
    return float(result.pvalue)


def pilot_go_no_go(summary: pd.DataFrame) -> dict[str, object]:
    matched = summary[summary["is_matched"] == 1]
    control = summary[summary["condition"].isin(["no_cue", "generic_neutral_bundle", "exact_repetition"])]
    control_max = (
        control.groupby(["task_name"], dropna=False)["accuracy"].max().reset_index().rename(
            columns={"accuracy": "best_control_accuracy"}
        )
    )
    comparison = matched.merge(control_max, on="task_name", how="left")
    comparison["matched_beats_controls"] = (
        comparison["accuracy"] > comparison["best_control_accuracy"]
    ).astype(int)
    diagonal_count = int(comparison["matched_beats_controls"].sum())

    concept_only = summary[summary["condition_type"] == "concept_bundle"]
    diagonal_advantage = (
        concept_only.groupby("task_name")
        .apply(
            lambda df: float(
                df.loc[df["is_matched"] == 1, "accuracy"].mean()
                - df.loc[df["is_matched"] == 0, "accuracy"].mean()
            )
        )
        .reset_index(name="diagonal_advantage")
    )
    positive_diagonal = int((diagonal_advantage["diagonal_advantage"] > 0).sum())

    go = diagonal_count >= 3 and positive_diagonal >= 4
    return {
        "go": go,
        "num_tasks_matched_beats_controls": diagonal_count,
        "num_tasks_positive_diagonal_advantage": positive_diagonal,
        "comparison": comparison,
        "diagonal_advantage": diagonal_advantage,
    }
