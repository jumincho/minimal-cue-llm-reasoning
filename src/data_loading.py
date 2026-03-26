from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .utils import PROJECT_ROOT, ensure_dir, read_jsonl, write_jsonl


TASK_TO_CONCEPT = {
    "causal_judgement": "causality",
    "date_understanding": "temporal",
    "logical_deduction_five_objects": "ordering",
    "tracking_shuffled_objects_five_objects": "state_tracking",
    "boolean_expressions": "boolean_logic",
    "web_of_lies": "consistency_truth",
    "formal_fallacies": "consistency_truth",
}

TASK_TO_FAMILY_V2 = {
    "causal_judgement": "causal_reasoning",
    "date_understanding": "temporal_reasoning",
    "logical_deduction_five_objects": "ordering_constraints",
    "tracking_shuffled_objects_five_objects": "state_tracking",
    "boolean_expressions": "boolean_logic",
    "web_of_lies": "truth_consistency",
    "formal_fallacies": "truth_consistency",
}

OPTION_RE = re.compile(r"^\(([A-Z])\)\s*(.+)$", re.MULTILINE)
BULLET_RE = re.compile(r"^-\s*(.+)$", re.MULTILINE)


@dataclass
class Candidate:
    label: str
    text: str
    score_text: str


@dataclass
class TaskExample:
    task_name: str
    concept_name: str
    item_id: str
    question: str
    answer_type: str
    gold_label: str
    gold_text: str
    candidates: list[Candidate]
    metadata: dict | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidates"] = [asdict(candidate) for candidate in self.candidates]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "TaskExample":
        candidates = [Candidate(**candidate) for candidate in payload["candidates"]]
        return cls(
            task_name=payload["task_name"],
            concept_name=payload["concept_name"],
            item_id=payload["item_id"],
            question=payload["question"],
            answer_type=payload["answer_type"],
            gold_label=payload["gold_label"],
            gold_text=payload["gold_text"],
            candidates=candidates,
            metadata=payload.get("metadata"),
        )


def repo_root(source_repo: str | Path) -> Path:
    path = Path(source_repo)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def maybe_clone_bbh(source_repo: str | Path) -> Path:
    root = repo_root(source_repo)
    if (root / "bbh").exists():
        return root
    ensure_dir(root.parent)
    raise FileNotFoundError(
        f"Expected BBH repo at {root}. Clone it before running the evaluation."
    )


def normalize_target(target: str) -> str:
    target = target.strip()
    match = re.fullmatch(r"\(([A-Z])\)", target)
    if match:
        return match.group(1)
    return target


def parse_candidates(input_text: str, target: str) -> tuple[str, str, list[Candidate]]:
    normalized_target = normalize_target(target)
    option_matches = OPTION_RE.findall(input_text)
    if option_matches:
        candidates = [
            Candidate(label=label, text=text.strip(), score_text=label)
            for label, text in option_matches
        ]
        gold_text = next(
            candidate.text for candidate in candidates if candidate.label == normalized_target
        )
        return "option_letter", gold_text, candidates

    bullet_matches = [entry.strip() for entry in BULLET_RE.findall(input_text)]
    if bullet_matches:
        candidates = [
            Candidate(label=text, text=text, score_text=text)
            for text in bullet_matches
        ]
        return "yes_no", normalized_target, candidates

    if normalized_target in {"True", "False"}:
        candidates = [
            Candidate(label="True", text="True", score_text="True"),
            Candidate(label="False", text="False", score_text="False"),
        ]
        return "true_false", normalized_target, candidates

    if normalized_target in {"Yes", "No"}:
        candidates = [
            Candidate(label="Yes", text="Yes", score_text="Yes"),
            Candidate(label="No", text="No", score_text="No"),
        ]
        return "yes_no", normalized_target, candidates

    raise ValueError(f"Could not parse candidates for target={target!r}")


def load_bbh_task(source_repo: str | Path, task_name: str) -> list[TaskExample]:
    root = maybe_clone_bbh(source_repo)
    task_path = root / "bbh" / f"{task_name}.json"
    payload = json.load(task_path.open("r", encoding="utf-8"))
    concept_name = TASK_TO_CONCEPT[task_name]
    rows: list[TaskExample] = []
    for idx, example in enumerate(payload["examples"]):
        answer_type, gold_text, candidates = parse_candidates(
            example["input"], example["target"]
        )
        normalized_target = normalize_target(example["target"])
        rows.append(
            TaskExample(
                task_name=task_name,
                concept_name=concept_name,
                item_id=f"{task_name}:{idx:04d}",
                question=example["input"].strip(),
                answer_type=answer_type,
                gold_label=normalized_target,
                gold_text=gold_text,
                candidates=candidates,
                metadata={
                    "family_v2": TASK_TO_FAMILY_V2[task_name],
                    "source": "bbh",
                },
            )
        )
    return rows


def load_bbh_tasks(
    source_repo: str | Path,
    tasks: Iterable[str],
    max_examples_per_task: int | None = None,
    seed: int = 7,
) -> list[TaskExample]:
    rng = random.Random(seed)
    selected: list[TaskExample] = []
    for task_name in tasks:
        examples = load_bbh_task(source_repo=source_repo, task_name=task_name)
        if max_examples_per_task is not None and len(examples) > max_examples_per_task:
            examples = rng.sample(examples, max_examples_per_task)
            examples.sort(key=lambda row: row.item_id)
        selected.extend(examples)
    return selected


def load_task_examples_jsonl(path_like: str | Path) -> list[TaskExample]:
    return [TaskExample.from_dict(row) for row in read_jsonl(path_like)]


def save_task_examples_jsonl(examples: Iterable[TaskExample], path_like: str | Path) -> None:
    write_jsonl([example.to_dict() for example in examples], path_like)
