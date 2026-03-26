from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .data_loading import TaskExample
from .utils import load_yaml


CONDITION_TO_BUNDLE = {
    "concept_bundle_causality": "causality",
    "concept_bundle_temporal": "temporal",
    "concept_bundle_ordering": "ordering",
    "concept_bundle_state_tracking": "state_tracking",
    "concept_bundle_boolean_logic": "boolean_logic",
    "concept_bundle_consistency_truth": "consistency_truth",
}


@dataclass
class PromptArtifacts:
    condition: str
    bundle_name: str | None
    cue_text: str
    answer_format: str
    prompt_text: str


def load_prompt_assets(
    bundles_path: str,
    templates_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundles = load_yaml(bundles_path)
    templates = load_yaml(templates_path)
    return bundles, templates


def answer_format_key(example: TaskExample) -> str:
    if example.answer_type == "option_letter":
        return "option_letter"
    if example.answer_type == "true_false":
        return "true_false"
    return "yes_no"


def bundle_for_condition(condition: str, task_concept: str) -> str | None:
    if condition == "no_cue":
        return None
    if condition == "generic_neutral_bundle":
        return "generic_neutral_bundle"
    if condition == "exact_repetition":
        return task_concept
    return CONDITION_TO_BUNDLE[condition]


def cue_text_for_condition(
    condition: str,
    task_concept: str,
    bundle_assets: dict[str, Any],
) -> tuple[str, str | None]:
    separator = bundle_assets["metadata"]["separator"]
    bundles = bundle_assets["bundles"]
    bundle_name = bundle_for_condition(condition=condition, task_concept=task_concept)
    if bundle_name is None:
        return "", None
    if condition == "exact_repetition":
        canonical = bundles[bundle_name]["canonical"]
        count = len(bundles[bundle_name]["variants"])
        return separator.join([canonical] * count), bundle_name
    if bundle_name == "generic_neutral_bundle":
        return separator.join(bundles[bundle_name]["variants"]), bundle_name
    return separator.join(bundles[bundle_name]["variants"]), bundle_name


def build_prompt_artifacts(
    example: TaskExample,
    condition: str,
    tokenizer,
    bundle_assets: dict[str, Any],
    template_assets: dict[str, Any],
) -> PromptArtifacts:
    cue_text, bundle_name = cue_text_for_condition(
        condition=condition,
        task_concept=example.concept_name,
        bundle_assets=bundle_assets,
    )
    format_key = answer_format_key(example)
    answer_format = template_assets["answer_formats"][format_key]
    user_text = template_assets["user_template"].format(
        cue_block=cue_text,
        answer_format=answer_format,
        question=example.question,
    )
    messages = [
        {"role": "system", "content": template_assets["system_prompt"].strip()},
        {"role": "user", "content": user_text.strip()},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return PromptArtifacts(
        condition=condition,
        bundle_name=bundle_name,
        cue_text=cue_text,
        answer_format=answer_format,
        prompt_text=prompt_text,
    )


def measure_condition_token_counts(
    tokenizer,
    bundle_assets: dict[str, Any],
    concepts: list[str],
) -> list[dict[str, Any]]:
    separator = bundle_assets["metadata"]["separator"]
    bundles = bundle_assets["bundles"]
    rows: list[dict[str, Any]] = []

    generic_text = separator.join(bundles["generic_neutral_bundle"]["variants"])
    rows.append(
        {
            "condition": "generic_neutral_bundle",
            "concept": "shared",
            "cue_text": generic_text,
            "token_count": len(tokenizer(generic_text, add_special_tokens=False)["input_ids"]),
        }
    )

    for concept in concepts:
        concept_bundle = separator.join(bundles[concept]["variants"])
        exact_bundle = separator.join(
            [bundles[concept]["canonical"]] * len(bundles[concept]["variants"])
        )
        rows.append(
            {
                "condition": "matched_concept_bundle",
                "concept": concept,
                "cue_text": concept_bundle,
                "token_count": len(
                    tokenizer(concept_bundle, add_special_tokens=False)["input_ids"]
                ),
            }
        )
        rows.append(
            {
                "condition": "exact_repetition",
                "concept": concept,
                "cue_text": exact_bundle,
                "token_count": len(
                    tokenizer(exact_bundle, add_special_tokens=False)["input_ids"]
                ),
            }
        )
    return rows
