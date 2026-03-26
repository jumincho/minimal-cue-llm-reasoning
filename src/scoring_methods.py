from __future__ import annotations

import json
import math
import random
import re
import string
from dataclasses import asdict, dataclass
from typing import Any

from .data_loading import Candidate, TaskExample
from .prompt_building import cue_text_for_condition
from .score_mc import MultipleChoiceScorer, ScoredCandidate
from .utils import normalize_text, stable_int_from_text


LETTER_LABELS = list(string.ascii_uppercase)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
V2_FAMILIES = [
    "boolean_logic",
    "ordering_constraints",
    "state_tracking",
    "temporal_reasoning",
    "truth_consistency",
    "causal_reasoning",
]
V1_TO_V2 = {
    "boolean_logic": "boolean_logic",
    "ordering": "ordering_constraints",
    "state_tracking": "state_tracking",
    "temporal": "temporal_reasoning",
    "consistency_truth": "truth_consistency",
    "causality": "causal_reasoning",
}


@dataclass
class PreparedCandidate:
    output_label: str
    answer_text: str
    score_text: str
    original_label: str
    original_text: str
    is_gold: bool


@dataclass
class PreparedExample:
    task_name: str
    family_name: str
    item_id: str
    condition: str
    cue_family: str | None
    cue_type: str | None
    cue_text: str
    wrapper_mode: str
    cue_placement: str
    prompt_text: str
    null_prompt_text: str
    prompt_question_text: str
    null_question_text: str
    answer_format: str
    option_mapping: dict[str, str]
    answer_labels: list[str]
    gold_label: str
    gold_text: str
    lexical_overlap_question: float
    lexical_overlap_answers: float
    candidates: list[PreparedCandidate]


def normalize_generation_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    cleaned = stripped.splitlines()[0].strip()
    cleaned = cleaned.strip().strip(".:,;()[]{}")
    return cleaned


def simple_tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def lexical_overlap_stats(cue_text: str, question_text: str, answer_texts: list[str]) -> tuple[float, float]:
    cue_tokens = simple_tokens(cue_text)
    if not cue_tokens:
        return 0.0, 0.0
    question_tokens = simple_tokens(question_text)
    answer_tokens = simple_tokens(" ".join(answer_texts))
    return (
        len(cue_tokens & question_tokens) / len(cue_tokens),
        len(cue_tokens & answer_tokens) / len(cue_tokens),
    )


def split_question_stem(question_text: str) -> str:
    if "\nOptions:" in question_text:
        return question_text.split("\nOptions:", 1)[0].strip()
    return question_text.strip()


def render_options_block(candidates: list[Candidate], label_style: str) -> str:
    lines = []
    if label_style == "letter":
        lines = [f"({candidate.label}) {candidate.text}" for candidate in candidates]
    else:
        lines = [f"- {candidate.text}" for candidate in candidates]
    return "\n".join(lines)


def render_user_prompt(
    template_assets: dict[str, Any],
    cue_text: str,
    answer_format: str,
    question: str,
    cue_placement: str,
) -> str:
    cue_line = f"Optional lens: {cue_text}" if cue_text else ""
    task_line = "Solve the following reasoning problem."
    answer_line = f"Answer with only {answer_format}."
    parts: list[str] = []
    if cue_placement == "before_task_instruction":
        parts = [cue_line, task_line, answer_line, "", question]
    elif cue_placement == "after_instruction":
        parts = [task_line, cue_line, answer_line, "", question]
    elif cue_placement == "before_question":
        parts = [task_line, answer_line, cue_line, "", question]
    elif cue_placement == "after_question":
        parts = [task_line, question, cue_line, answer_line]
    else:
        raise ValueError(f"Unsupported cue placement: {cue_placement}")
    if cue_placement != "after_question":
        rendered = "\n".join(part for part in parts if part != "")
    else:
        rendered = "\n\n".join(part for part in parts if part)
    return rendered.strip()


def build_chat_prompt(
    tokenizer,
    template_assets: dict[str, Any],
    cue_text: str,
    answer_format: str,
    question: str,
    cue_placement: str = "after_instruction",
) -> str:
    user_text = render_user_prompt(
        template_assets=template_assets,
        cue_text=cue_text,
        answer_format=answer_format,
        question=question,
        cue_placement=cue_placement,
    )
    messages = [
        {"role": "system", "content": template_assets["system_prompt"].strip()},
        {"role": "user", "content": user_text.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def family_from_example(example: TaskExample) -> str:
    if example.metadata and example.metadata.get("family_v2"):
        return example.metadata["family_v2"]
    return V1_TO_V2.get(example.concept_name, example.concept_name)


def resolve_old_cue(
    example: TaskExample,
    condition: str,
    bundle_assets: dict[str, Any],
) -> tuple[str, str | None, str | None]:
    cue_text, bundle_name = cue_text_for_condition(
        condition=condition,
        task_concept=example.concept_name,
        bundle_assets=bundle_assets,
    )
    cue_type = None
    if condition == "generic_neutral_bundle":
        cue_type = "generic_neutral"
    elif condition == "exact_repetition":
        cue_type = "exact_repetition"
    elif condition.startswith("concept_bundle_"):
        cue_type = "semantic"
    return cue_text, bundle_name, cue_type


def resolve_v2_cue(
    example: TaskExample,
    condition: str,
    bundle_assets: dict[str, Any],
    off_target_family: str | None = None,
) -> tuple[str, str | None, str | None]:
    separator = bundle_assets["metadata"]["separator"]
    family_name = family_from_example(example)
    controls = bundle_assets["controls"]
    families = bundle_assets["families"]
    if condition == "no_cue":
        return "", None, None
    if condition in {"generic_neutral", "generic_neutral_bundle"}:
        cue = controls["generic_neutral"]["text"]
        return separator.join(cue), "generic_neutral", "generic_neutral"
    if condition == "single_canonical":
        canonical = families[family_name]["canonical"]
        return canonical, family_name, "canonical_semantic"
    if condition == "exact_repetition":
        canonical = families[family_name]["canonical"]
        return separator.join([canonical] * 4), family_name, "exact_repetition"
    target_family = family_name
    if condition in {"off_target_semantic", "off_target_procedural", "off_target_mixed"}:
        if off_target_family is None:
            raise ValueError("off_target_family is required for off-target cue conditions")
        target_family = off_target_family
    cue_type_map = {
        "matched_semantic": "semantic",
        "matched_procedural": "procedural",
        "matched_mixed": "mixed",
        "matched_lexical_overlap": "lexical_overlap",
        "off_target_semantic": "semantic",
        "off_target_procedural": "procedural",
        "off_target_mixed": "mixed",
        "matched_near_miss": "near_miss",
    }
    cue_type = cue_type_map[condition]
    cue = families[target_family][cue_type]["text"]
    return separator.join(cue), target_family, cue_type


def randomized_candidates(example: TaskExample, seed: int) -> tuple[list[PreparedCandidate], dict[str, str], str]:
    shuffled = list(example.candidates)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    letters = LETTER_LABELS[: len(shuffled)]
    prepared: list[PreparedCandidate] = []
    option_mapping = {}
    gold_output_label = ""
    for letter, candidate in zip(letters, shuffled, strict=True):
        option_mapping[letter] = candidate.text
        is_gold = candidate.label == example.gold_label
        if is_gold:
            gold_output_label = letter
        prepared.append(
            PreparedCandidate(
                output_label=letter,
                answer_text=candidate.text,
                score_text=letter,
                original_label=candidate.label,
                original_text=candidate.text,
                is_gold=is_gold,
            )
        )
    return prepared, option_mapping, gold_output_label


def direct_candidates(example: TaskExample) -> tuple[list[PreparedCandidate], dict[str, str], str]:
    prepared = []
    option_mapping = {}
    gold_output_label = example.gold_label
    for candidate in example.candidates:
        option_mapping[candidate.label] = candidate.text
        prepared.append(
            PreparedCandidate(
                output_label=candidate.label,
                answer_text=candidate.text,
                score_text=candidate.score_text,
                original_label=candidate.label,
                original_text=candidate.text,
                is_gold=candidate.label == example.gold_label,
            )
        )
    return prepared, option_mapping, gold_output_label


def option_text_candidates(example: TaskExample) -> tuple[list[PreparedCandidate], dict[str, str], str]:
    prepared = []
    option_mapping = {}
    gold_output_label = example.gold_text
    for candidate in example.candidates:
        output_label = candidate.text
        option_mapping[output_label] = candidate.text
        prepared.append(
            PreparedCandidate(
                output_label=output_label,
                answer_text=candidate.text,
                score_text=candidate.text,
                original_label=candidate.label,
                original_text=candidate.text,
                is_gold=candidate.label == example.gold_label,
            )
        )
    return prepared, option_mapping, gold_output_label


def direct_null_question(example: TaskExample, template_assets: dict[str, Any]) -> str:
    stem = template_assets["null_stem"]
    if example.answer_type == "option_letter":
        options = render_options_block(example.candidates, label_style="letter")
        return template_assets["wrapped_question_template"].format(stem=stem, options_block=options).strip()
    if example.answer_type in {"yes_no", "true_false"}:
        options = render_options_block(example.candidates, label_style="bullet")
        return template_assets["wrapped_question_template"].format(stem=stem, options_block=options).strip()
    return stem


def prepare_evaluation_example(
    example: TaskExample,
    tokenizer,
    template_assets: dict[str, Any],
    cue_text: str,
    condition: str,
    cue_family: str | None,
    cue_type: str | None,
    wrapper_mode: str,
    seed: int,
    cue_placement: str = "after_instruction",
) -> PreparedExample:
    family_name = family_from_example(example)
    answer_format_key = "option_letter" if wrapper_mode == "randomized_letters" else (
        "option_text" if wrapper_mode == "option_text" else (
        "option_letter" if example.answer_type == "option_letter" else (
            "true_false" if example.answer_type == "true_false" else "yes_no"
        )
    ))
    answer_format = template_assets["answer_formats"][answer_format_key]
    if wrapper_mode == "randomized_letters":
        candidate_seed = stable_int_from_text(f"{seed}:{example.item_id}:{condition}:{wrapper_mode}")
        prepared_candidates, option_mapping, gold_output_label = randomized_candidates(example, candidate_seed)
        stem = split_question_stem(example.question)
        options_block = "\n".join(
            f"({candidate.output_label}) {candidate.answer_text}"
            for candidate in prepared_candidates
        )
        question_text = template_assets["wrapped_question_template"].format(
            stem=stem,
            options_block=options_block,
        ).strip()
        null_question = template_assets["wrapped_question_template"].format(
            stem=template_assets["null_stem"],
            options_block=options_block,
        ).strip()
        gold_text = option_mapping[gold_output_label]
        answer_labels = list(option_mapping.keys())
    elif wrapper_mode == "option_text":
        prepared_candidates, option_mapping, gold_output_label = option_text_candidates(example)
        question_text = example.question
        null_question = direct_null_question(example, template_assets)
        gold_text = example.gold_text
        answer_labels = [candidate.output_label for candidate in prepared_candidates]
    else:
        prepared_candidates, option_mapping, gold_output_label = direct_candidates(example)
        question_text = example.question
        null_question = direct_null_question(example, template_assets)
        gold_text = example.gold_text
        answer_labels = [candidate.output_label for candidate in prepared_candidates]

    prompt_text = build_chat_prompt(
        tokenizer=tokenizer,
        template_assets=template_assets,
        cue_text=cue_text,
        answer_format=answer_format,
        question=question_text,
        cue_placement=cue_placement,
    )
    null_prompt_text = build_chat_prompt(
        tokenizer=tokenizer,
        template_assets=template_assets,
        cue_text=cue_text,
        answer_format=answer_format,
        question=null_question,
        cue_placement=cue_placement,
    )
    question_overlap, answer_overlap = lexical_overlap_stats(
        cue_text=cue_text,
        question_text=question_text,
        answer_texts=[candidate.answer_text for candidate in prepared_candidates],
    )
    return PreparedExample(
        task_name=example.task_name,
        family_name=family_name,
        item_id=example.item_id,
        condition=condition,
        cue_family=cue_family,
        cue_type=cue_type,
        cue_text=cue_text,
        wrapper_mode=wrapper_mode,
        cue_placement=cue_placement,
        prompt_text=prompt_text,
        null_prompt_text=null_prompt_text,
        prompt_question_text=question_text,
        null_question_text=null_question,
        answer_format=answer_format,
        option_mapping=option_mapping,
        answer_labels=answer_labels,
        gold_label=gold_output_label,
        gold_text=gold_text,
        lexical_overlap_question=question_overlap,
        lexical_overlap_answers=answer_overlap,
        candidates=prepared_candidates,
    )


def candidate_value(candidate: ScoredCandidate, mode: str) -> float:
    if mode == "raw_next_token":
        return candidate.first_token_logprob
    if mode == "full_length_norm":
        return candidate.length_norm_logprob
    if mode == "full_logprob_sum":
        return candidate.logprob_sum
    raise ValueError(f"Unsupported score mode: {mode}")


def softmax_map(score_map: dict[str, float]) -> dict[str, float]:
    if not score_map:
        return {}
    max_score = max(score_map.values())
    weights = {key: math.exp(value - max_score) for key, value in score_map.items()}
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def score_prepared_examples(
    scorer: MultipleChoiceScorer,
    prepared_examples: list[PreparedExample],
    scoring_method: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    prompt_texts = [example.prompt_text for example in prepared_examples]
    candidate_payloads = [
        [
            {
                "label": candidate.output_label,
                "text": candidate.answer_text,
                "score_text": candidate.score_text,
            }
            for candidate in example.candidates
        ]
        for example in prepared_examples
    ]
    direct_mode = scoring_method if scoring_method != "calibrated_full_length_norm" else "full_length_norm"
    scored = scorer.score_batch(prompt_texts, candidate_payloads, batch_size=batch_size)
    null_scored = None
    if scoring_method == "calibrated_full_length_norm":
        null_prompts = [example.null_prompt_text for example in prepared_examples]
        null_scored = scorer.score_batch(null_prompts, candidate_payloads, batch_size=batch_size)

    rows: list[dict[str, Any]] = []
    for idx, (prepared, candidates) in enumerate(zip(prepared_examples, scored, strict=True)):
        score_map = {candidate.label: candidate_value(candidate, direct_mode) for candidate in candidates}
        null_score_map = {}
        if null_scored is not None:
            null_score_map = {
                candidate.label: candidate_value(candidate, "full_length_norm")
                for candidate in null_scored[idx]
            }
            score_map = {
                label: score_map[label] - null_score_map[label]
                for label in score_map
            }
        ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
        predicted_label = ranked[0][0]
        best_other = ranked[1][1] if len(ranked) > 1 else float("-inf")
        candidate_probs = softmax_map(score_map)
        null_candidate_probs = softmax_map(null_score_map) if null_score_map else {}
        rows.append(
            {
                "task_name": prepared.task_name,
                "family_name": prepared.family_name,
                "item_id": prepared.item_id,
                "condition": prepared.condition,
                "cue_family": prepared.cue_family,
                "cue_type": prepared.cue_type,
                "matched": int(
                    prepared.cue_family is not None and prepared.cue_family == prepared.family_name
                ),
                "gold": prepared.gold_label,
                "gold_text": prepared.gold_text,
                "pred": predicted_label,
                "pred_text": prepared.option_mapping[predicted_label],
                "correct": int(predicted_label == prepared.gold_label),
                "confidence_margin": float(score_map[prepared.gold_label] - best_other),
                "answer_labels": json.dumps(prepared.answer_labels),
                "option_mapping": json.dumps(prepared.option_mapping, ensure_ascii=True),
                "prompt_text": prepared.prompt_text,
                "null_prompt_text": prepared.null_prompt_text,
                "prompt_question_text": prepared.prompt_question_text,
                "cue_text": prepared.cue_text,
                "cue_placement": prepared.cue_placement,
                "scoring_method": scoring_method,
                "wrapper_mode": prepared.wrapper_mode,
                "candidate_scores": json.dumps(score_map, ensure_ascii=True),
                "candidate_probabilities": json.dumps(candidate_probs, ensure_ascii=True),
                "null_candidate_probabilities": json.dumps(null_candidate_probs, ensure_ascii=True),
                "lexical_overlap_question": prepared.lexical_overlap_question,
                "lexical_overlap_answers": prepared.lexical_overlap_answers,
            }
        )
    return rows


def parse_generated_prediction(generated_text: str, prepared: PreparedExample) -> str | None:
    normalized = normalize_generation_text(generated_text)
    if not normalized:
        return None
    if prepared.wrapper_mode == "randomized_letters":
        first = normalized[:1].upper()
        if first in prepared.option_mapping:
            return first
    normalized_lower = normalize_text(normalized).lower()
    for candidate in prepared.candidates:
        if normalized_lower == candidate.output_label.lower():
            return candidate.output_label
        if normalized_lower == candidate.answer_text.lower():
            return candidate.output_label
    for candidate in prepared.candidates:
        if normalized_lower.startswith(candidate.output_label.lower()):
            return candidate.output_label
        if normalized_lower.startswith(candidate.answer_text.lower()):
            return candidate.output_label
    return None


def constrained_generation_rows(
    scorer: MultipleChoiceScorer,
    prepared_examples: list[PreparedExample],
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    generations = scorer.generate_batch(
        [example.prompt_text for example in prepared_examples],
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    rows = []
    for prepared, generated_text in zip(prepared_examples, generations, strict=True):
        predicted_label = parse_generated_prediction(generated_text, prepared)
        rows.append(
            {
                "task_name": prepared.task_name,
                "family_name": prepared.family_name,
                "item_id": prepared.item_id,
                "condition": prepared.condition,
                "cue_family": prepared.cue_family,
                "cue_type": prepared.cue_type,
                "matched": int(
                    prepared.cue_family is not None and prepared.cue_family == prepared.family_name
                ),
                "gold": prepared.gold_label,
                "gold_text": prepared.gold_text,
                "pred": predicted_label or "UNPARSED",
                "pred_text": prepared.option_mapping.get(predicted_label or "", "UNPARSED"),
                "correct": int(predicted_label == prepared.gold_label),
                "confidence_margin": None,
                "answer_labels": json.dumps(prepared.answer_labels),
                "option_mapping": json.dumps(prepared.option_mapping, ensure_ascii=True),
                "prompt_text": prepared.prompt_text,
                "null_prompt_text": prepared.null_prompt_text,
                "prompt_question_text": prepared.prompt_question_text,
                "cue_text": prepared.cue_text,
                "cue_placement": prepared.cue_placement,
                "scoring_method": "constrained_generation",
                "wrapper_mode": prepared.wrapper_mode,
                "candidate_scores": json.dumps({}, ensure_ascii=True),
                "candidate_probabilities": json.dumps({}, ensure_ascii=True),
                "null_candidate_probabilities": json.dumps({}, ensure_ascii=True),
                "generated_text": generated_text,
                "lexical_overlap_question": prepared.lexical_overlap_question,
                "lexical_overlap_answers": prepared.lexical_overlap_answers,
            }
        )
    return rows


def candidate_rows_from_example(prepared: PreparedExample) -> list[dict[str, Any]]:
    return [asdict(candidate) for candidate in prepared.candidates]
