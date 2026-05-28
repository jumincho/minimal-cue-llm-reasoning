"""The runner: take a model + the consolidated test set, run every eval mode.

Step (3) in the rerun flow. Loads a per-model config (see
`configs/confirmatory_v3_<model>.yaml`), then for each TaskExample evaluates
every condition under every evaluation mode (`free_form_only`,
`cot_before_options`, `standard_mc`, `binding_only`).

The four-mode core is the **decoupled evaluation** at the heart of the
project: it lets us measure *where* a cue moves the model — at the
solve step, the binding step, or both. The closure report reports
`solve = standard_mc → binding_only` and `final = standard_mc`.

Writes per-item raw rows under `results/raw/confirmatory_v3/<model>_items.jsonl`
and per-mode CSV summaries under `results/processed/confirmatory_v3/per_model/`.

The `config_hash` is embedded in each row so a downstream postprocess can
flag stale runs vs. a changed config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import string
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .data_loading import TaskExample, load_task_examples_jsonl
from .prompt_building import load_prompt_assets
from .score_mc import MultipleChoiceScorer
from .scoring_methods import family_from_example, resolve_v2_cue
from .utils import ensure_dir, load_yaml, normalize_text, set_seed


LETTER_LABELS = list(string.ascii_uppercase)
LETTER_RE = re.compile(r"\b([A-Z])\b")
FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE)
ROLE_LINE_RE = re.compile(r"^\s*assistant\s*:?\s*$", re.IGNORECASE)
BINARY_RE = re.compile(r"\b(true|false|yes|no)\b", re.IGNORECASE)


@dataclass
class OptionPack:
    question_with_options: str
    option_mapping: dict[str, str]
    gold_letter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--synthetic-path", default="")
    parser.add_argument("--results-suffix", default="")
    return parser.parse_args()


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def append_suffix_to_path(path_text: str, suffix: str) -> str:
    path = Path(path_text)
    if not suffix:
        return path_text
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix}"))


def resolve_batch_size(batch_size_config: Any, evaluation_mode: str) -> int:
    if isinstance(batch_size_config, dict):
        if evaluation_mode in batch_size_config:
            return int(batch_size_config[evaluation_mode])
        if "default" in batch_size_config:
            return int(batch_size_config["default"])
        raise KeyError(f"Missing batch size for evaluation mode: {evaluation_mode}")
    return int(batch_size_config)


def split_question_stem(question_text: str) -> str:
    if "\nOptions:" in question_text:
        return question_text.split("\nOptions:", 1)[0].strip()
    return question_text.strip()


def normalize_key(text: str) -> str:
    cleaned = normalize_text(text).strip().strip(".:,;()[]{}\"'")
    return cleaned.casefold()


def meaningful_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def candidate_line_variants(text: str) -> list[str]:
    lines = meaningful_lines(text)
    variants: list[str] = []
    for line in lines:
        if ROLE_LINE_RE.match(line):
            continue
        variants.append(line)
    answer = extract_final_answer_segment(text)
    if answer:
        variants.append(answer)
    return variants


def extract_candidate_match(text: str, candidates: list[str]) -> str | None:
    normalized_candidates = {normalize_key(candidate): candidate for candidate in candidates}
    for variant in reversed(candidate_line_variants(text)):
        normalized_answer = normalize_key(variant)
        if not normalized_answer:
            continue
        if normalized_answer in normalized_candidates:
            return normalized_candidates[normalized_answer]
        single_hits = [
            candidate
            for candidate in candidates
            if normalize_key(candidate) in normalized_answer
        ]
        if len(single_hits) == 1:
            return single_hits[0]
    return None


def extract_final_answer_segment(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    matches = FINAL_ANSWER_RE.findall(stripped)
    if matches:
        stripped = matches[-1].strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not lines:
        return stripped.strip().strip(".:,;()[]{}\"'")
    for line in reversed(lines):
        if ROLE_LINE_RE.match(line):
            continue
        return line.strip().strip(".:,;()[]{}\"'")
    return lines[-1].strip().strip(".:,;()[]{}\"'")


def extract_binary_from_line(answer_type: str, line: str) -> str | None:
    lowered = line.casefold()
    if answer_type == "true_false":
        valid = {"true": "True", "false": "False"}
    elif answer_type == "yes_no":
        valid = {"yes": "Yes", "no": "No"}
    else:
        raise ValueError(f"Unsupported binary answer type: {answer_type}")

    cleaned = line.strip().strip(".:,;()[]{}\"'").casefold()
    if cleaned in valid:
        return valid[cleaned]

    matches = [match.group(1).casefold() for match in BINARY_RE.finditer(line)]
    matches = [match for match in matches if match in valid]
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) == 1:
        if "answer with only" in lowered:
            return None
        return valid[unique_matches[0]]
    return None


def parse_binary_answer(answer_type: str, text: str) -> str | None:
    if answer_type not in {"true_false", "yes_no"}:
        raise ValueError(f"Unsupported binary answer type: {answer_type}")
    for line in reversed(candidate_line_variants(text)):
        parsed = extract_binary_from_line(answer_type, line)
        if parsed is not None:
            return parsed
    return None


def parse_free_form_answer(example: TaskExample, text: str) -> str | None:
    if example.answer_type in {"true_false", "yes_no"}:
        return parse_binary_answer(example.answer_type, text)
    candidates = [candidate.text for candidate in example.candidates]
    return extract_candidate_match(text, candidates)


def parse_letter_answer(text: str, valid_letters: set[str]) -> str | None:
    for line in reversed(candidate_line_variants(text)):
        cleaned = line.strip().strip(".:,;()[]{}")
        if len(cleaned) == 1 and cleaned in valid_letters:
            return cleaned
        if "option" in line.casefold() and "answer with only" in line.casefold():
            continue
        matches = [match.group(1) for match in LETTER_RE.finditer(cleaned) if match.group(1) in valid_letters]
        unique_matches = list(dict.fromkeys(matches))
        if len(unique_matches) == 1:
            return unique_matches[0]
    return None


def randomized_option_pack(example: TaskExample, seed: int) -> OptionPack:
    import random

    rng = random.Random(seed)
    shuffled = list(example.candidates)
    rng.shuffle(shuffled)
    letters = LETTER_LABELS[: len(shuffled)]
    mapping = {letter: candidate.text for letter, candidate in zip(letters, shuffled, strict=True)}
    gold_letter = ""
    for letter, candidate in zip(letters, shuffled, strict=True):
        if candidate.text == example.gold_text:
            gold_letter = letter
            break
    if not gold_letter:
        raise ValueError(f"Could not locate gold candidate for {example.item_id}")
    options_block = "\n".join(f"({letter}) {mapping[letter]}" for letter in letters)
    question_with_options = f"{split_question_stem(example.question)}\nOptions:\n{options_block}"
    return OptionPack(
        question_with_options=question_with_options,
        option_mapping=mapping,
        gold_letter=gold_letter,
    )


def render_cue(cue_text: str, formatting_style: str) -> str:
    if not cue_text:
        return ""
    if formatting_style == "plain_sentence":
        return f"Optional lens: {cue_text}"
    if formatting_style == "bracketed_note":
        return f"[Optional lens: {cue_text}]"
    if formatting_style == "bullet_block":
        return "Reasoning lens:\n- " + "\n- ".join(part.strip() for part in cue_text.split("|"))
    raise ValueError(f"Unsupported formatting style: {formatting_style}")


def render_user_prompt(
    cue_text: str,
    cue_placement: str,
    formatting_style: str,
    task_instruction: str,
    question_text: str,
    answer_instruction: str,
    extra_block: str | None = None,
) -> str:
    cue_block = render_cue(cue_text, formatting_style)
    pieces: list[str] = []
    if cue_placement == "system_prelude":
        pieces = [cue_block, task_instruction, answer_instruction, extra_block or "", question_text]
    elif cue_placement == "before_task_instruction":
        pieces = [cue_block, task_instruction, answer_instruction, extra_block or "", question_text]
    elif cue_placement == "after_instruction":
        pieces = [task_instruction, cue_block, answer_instruction, extra_block or "", question_text]
    elif cue_placement == "before_question":
        pieces = [task_instruction, answer_instruction, cue_block, extra_block or "", question_text]
    elif cue_placement == "after_question":
        pieces = [task_instruction, question_text, cue_block, extra_block or "", answer_instruction]
    else:
        raise ValueError(f"Unsupported cue placement: {cue_placement}")
    return "\n\n".join(piece for piece in pieces if piece).strip()


def build_chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_free_form_prompt(
    tokenizer,
    system_prompt: str,
    cue_text: str,
    cue_placement: str,
    formatting_style: str,
    example: TaskExample,
) -> str:
    answer_instruction = "Give only the final answer text."
    if example.answer_type == "true_false":
        answer_instruction = "Answer with only True or False."
    if example.answer_type == "yes_no":
        answer_instruction = "Answer with only Yes or No."
    user_prompt = render_user_prompt(
        cue_text=cue_text,
        cue_placement=cue_placement,
        formatting_style=formatting_style,
        task_instruction="Solve the reasoning problem before you.",
        question_text=split_question_stem(example.question),
        answer_instruction=answer_instruction,
    )
    return build_chat_prompt(tokenizer, system_prompt=system_prompt, user_prompt=user_prompt)


def build_cot_solve_prompt(
    tokenizer,
    system_prompt: str,
    cue_text: str,
    cue_placement: str,
    formatting_style: str,
    example: TaskExample,
) -> str:
    if example.answer_type in {"true_false", "yes_no"}:
        final_spec = example.gold_text
    else:
        final_spec = "the final answer text"
    user_prompt = render_user_prompt(
        cue_text=cue_text,
        cue_placement=cue_placement,
        formatting_style=formatting_style,
        task_instruction="Reason through the problem before answering.",
        question_text=split_question_stem(example.question),
        answer_instruction=f"Think step by step briefly, then end with `Final answer: <answer>`. Use {final_spec}.",
    )
    return build_chat_prompt(tokenizer, system_prompt=system_prompt, user_prompt=user_prompt)


def build_standard_mc_prompt(
    tokenizer,
    system_prompt: str,
    cue_text: str,
    cue_placement: str,
    formatting_style: str,
    question_with_options: str,
) -> str:
    user_prompt = render_user_prompt(
        cue_text=cue_text,
        cue_placement=cue_placement,
        formatting_style=formatting_style,
        task_instruction="Solve the reasoning problem and select the matching option.",
        question_text=question_with_options,
        answer_instruction="Answer with only the option letter.",
    )
    return build_chat_prompt(tokenizer, system_prompt=system_prompt, user_prompt=user_prompt)


def build_binding_prompt(
    tokenizer,
    system_prompt: str,
    cue_text: str,
    cue_placement: str,
    formatting_style: str,
    question_with_options: str,
    provisional_answer: str,
) -> str:
    extra_block = f"Provisional answer: {provisional_answer}"
    user_prompt = render_user_prompt(
        cue_text=cue_text,
        cue_placement=cue_placement,
        formatting_style=formatting_style,
        task_instruction="Do not solve the problem again. Only bind the provisional answer to the correct option.",
        question_text=question_with_options,
        answer_instruction="Answer with only the option letter.",
        extra_block=extra_block,
    )
    return build_chat_prompt(tokenizer, system_prompt=system_prompt, user_prompt=user_prompt)


def rows_for_examples(
    examples: list[TaskExample],
    condition: str,
    cue_payloads: dict[str, tuple[str, str | None, str | None]],
    option_packs: dict[str, OptionPack],
    model_alias: str,
    model_id: str,
    eval_mode: str,
    cue_placement: str,
    formatting_style: str,
    seed: int,
    config_digest: str,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        cue_text, cue_family, cue_type = cue_payloads[example.item_id]
        option_pack = option_packs[example.item_id]
        rows.append(
            {
                "model_alias": model_alias,
                "model_id": model_id,
                "eval_seed": seed,
                "config_hash": config_digest,
                "task_name": example.task_name,
                "family_name": family_from_example(example),
                "item_id": example.item_id,
                "difficulty": example.metadata.get("difficulty", "unknown") if example.metadata else "unknown",
                "condition": condition,
                "cue_family": cue_family,
                "cue_type": cue_type,
                "evaluation_mode": eval_mode,
                "cue_placement": cue_placement,
                "formatting_style": formatting_style,
                "answer_type": example.answer_type,
                "question_stem": split_question_stem(example.question),
                "question_with_options": option_pack.question_with_options,
                "option_mapping": json.dumps(option_pack.option_mapping, ensure_ascii=True, sort_keys=True),
                "gold_text": example.gold_text,
                "gold_letter": option_pack.gold_letter,
                "cue_text": cue_text,
            }
        )
    return rows


def finalize_mode_summary(item_frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "model_alias",
        "evaluation_mode",
        "condition",
        "cue_type",
        "cue_family",
        "family_name",
        "cue_placement",
        "formatting_style",
    ]
    frame = item_frame.copy()
    for column in ["solve_correct", "binding_correct", "final_correct", "parse_failure_solve", "parse_failure_bind"]:
        if column not in frame.columns:
            frame[column] = pd.NA
    summary = (
        frame.groupby(group_cols, dropna=False)
        .agg(
            n=("item_id", "count"),
            final_accuracy=("final_correct", "mean"),
            solve_accuracy=("solve_correct", "mean"),
            binding_accuracy=("binding_correct", "mean"),
            solve_parse_failure=("parse_failure_solve", "mean"),
            bind_parse_failure=("parse_failure_bind", "mean"),
        )
        .reset_index()
    )
    conditional_rows = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        solved = group[group["solve_correct"] == 1]
        conditional = solved["binding_correct"].mean() if not solved.empty else pd.NA
        conditional_rows.append(list(keys) + [conditional])
    conditional_cols = group_cols + ["binding_accuracy_given_correct_solve"]
    conditional_frame = pd.DataFrame(conditional_rows, columns=conditional_cols)
    summary = summary.merge(conditional_frame, on=group_cols, how="left")
    return summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.synthetic_path:
        config["data"]["synthetic_path"] = args.synthetic_path
    if args.results_suffix:
        for key in ["raw_jsonl", "items_csv", "summary_csv", "family_summary_csv", "report_path"]:
            if key in config["results"]:
                config["results"][key] = append_suffix_to_path(config["results"][key], args.results_suffix)
    seed = int(config["experiment"]["seed"] if args.seed is None else args.seed)
    set_seed(seed)
    config_digest = config_hash(config)

    examples = load_task_examples_jsonl(config["data"]["synthetic_path"])
    bundles_v2, template_assets = load_prompt_assets(
        config["prompt"]["bundles_v2_path"],
        config["prompt"]["templates_path"],
    )
    model_payload = config["model"]
    scorer = MultipleChoiceScorer(
        model_id=model_payload["model_id"],
        gpu_id=args.gpu_id,
        load_in_4bit=bool(model_payload.get("load_in_4bit", True)),
        compute_dtype=model_payload.get("compute_dtype", "bfloat16"),
        attn_implementation=model_payload.get("attn_implementation"),
    )
    system_prompt = template_assets["system_prompt"].strip()
    cue_placement = config["evaluation"]["cue_placement"]
    formatting_style = config["evaluation"]["formatting_style"]

    option_packs = {
        example.item_id: randomized_option_pack(example, seed + idx * 13)
        for idx, example in enumerate(examples)
    }

    all_rows: list[dict[str, Any]] = []
    max_tokens = config["evaluation"]["max_new_tokens"]
    batch_size_config = config["evaluation"].get("batch_size", 8)

    def flush_outputs() -> None:
        if not all_rows:
            return
        item_frame = pd.DataFrame(all_rows)
        summary = finalize_mode_summary(item_frame)
        family_summary = summary.copy()
        aggregate_summary = (
            summary.groupby(
                [
                    "model_alias",
                    "evaluation_mode",
                    "condition",
                    "cue_type",
                    "cue_family",
                    "cue_placement",
                    "formatting_style",
                ],
                dropna=False,
            )
            .agg(
                n=("n", "sum"),
                final_accuracy=("final_accuracy", "mean"),
                solve_accuracy=("solve_accuracy", "mean"),
                binding_accuracy=("binding_accuracy", "mean"),
                binding_accuracy_given_correct_solve=("binding_accuracy_given_correct_solve", "mean"),
                solve_parse_failure=("solve_parse_failure", "mean"),
                bind_parse_failure=("bind_parse_failure", "mean"),
            )
            .reset_index()
        )

        outputs = config["results"]
        for key, frame in [
            ("items_csv", item_frame),
            ("summary_csv", aggregate_summary),
            ("family_summary_csv", family_summary),
        ]:
            if key in outputs:
                target = Path(outputs[key])
                ensure_dir(target.parent)
                frame.to_csv(target, index=False)
        if "raw_jsonl" in outputs:
            target = Path(outputs["raw_jsonl"])
            ensure_dir(target.parent)
            item_frame.to_json(target, orient="records", lines=True)
        if "report_path" in outputs:
            target = Path(outputs["report_path"])
            lines = [
                "# Phase B Decoupled Evaluation",
                "",
                f"- Model: `{model_payload['model_id']}`",
                f"- Seed: `{seed}`",
                f"- Config hash: `{config_digest}`",
                f"- Synthetic path: `{config['data']['synthetic_path']}`",
                f"- Conditions: {', '.join(config['evaluation']['conditions'])}",
                f"- Evaluation modes: {', '.join(config['evaluation']['evaluation_modes'])}",
                "",
                "## Aggregate Summary",
                "",
                "```text",
                aggregate_summary.to_string(index=False),
                "```",
            ]
            ensure_dir(target.parent)
            target.write_text("\n".join(lines), encoding="utf-8")

    for condition in config["evaluation"]["conditions"]:
        condition_start = time.time()
        print(
            f"[condition-start] model={model_payload['alias']} synthetic={config['data']['synthetic_path']} "
            f"condition={condition} placement={cue_placement} format={formatting_style}",
            flush=True,
        )
        cue_payloads = {}
        for example in examples:
            cue_payloads[example.item_id] = resolve_v2_cue(
                example=example,
                condition=condition,
                bundle_assets=bundles_v2,
            )

        if "free_form_only" in config["evaluation"]["evaluation_modes"]:
            prompts = [
                build_free_form_prompt(
                    tokenizer=scorer.tokenizer,
                    system_prompt=system_prompt,
                    cue_text=cue_payloads[example.item_id][0],
                    cue_placement=cue_placement,
                    formatting_style=formatting_style,
                    example=example,
                )
                for example in examples
            ]
            outputs = scorer.generate_batch(
                prompts,
                max_new_tokens=int(max_tokens["free_form_only"]),
                batch_size=resolve_batch_size(batch_size_config, "free_form_only"),
            )
            rows = rows_for_examples(
                examples=examples,
                condition=condition,
                cue_payloads=cue_payloads,
                option_packs=option_packs,
                model_alias=model_payload["alias"],
                model_id=model_payload["model_id"],
                eval_mode="free_form_only",
                cue_placement=cue_placement,
                formatting_style=formatting_style,
                seed=seed,
                config_digest=config_digest,
            )
            for row, example, prompt_text, output_text in zip(rows, examples, prompts, outputs, strict=True):
                parsed = parse_free_form_answer(example, output_text)
                row.update(
                    {
                        "solve_prompt_text": prompt_text,
                        "binding_prompt_text": "",
                        "solve_output_text": output_text,
                        "binding_output_text": "",
                        "provisional_answer": "",
                        "parsed_solve_answer": parsed,
                        "parsed_binding_letter": "",
                        "solve_correct": int(parsed == example.gold_text) if parsed is not None else 0,
                        "binding_correct": pd.NA,
                        "final_correct": int(parsed == example.gold_text) if parsed is not None else 0,
                        "parse_failure_solve": int(parsed is None),
                        "parse_failure_bind": pd.NA,
                    }
                )
            all_rows.extend(rows)

        if "cot_before_options" in config["evaluation"]["evaluation_modes"]:
            solve_prompts = [
                build_cot_solve_prompt(
                    tokenizer=scorer.tokenizer,
                    system_prompt=system_prompt,
                    cue_text=cue_payloads[example.item_id][0],
                    cue_placement=cue_placement,
                    formatting_style=formatting_style,
                    example=example,
                )
                for example in examples
            ]
            solve_outputs = scorer.generate_batch(
                solve_prompts,
                max_new_tokens=int(max_tokens["cot_before_options"]),
                batch_size=resolve_batch_size(batch_size_config, "cot_before_options"),
            )
            provisional_answers: list[str] = []
            solve_parsed: list[str | None] = []
            solve_correct: list[int] = []
            for example, output_text in zip(examples, solve_outputs, strict=True):
                parsed = parse_free_form_answer(example, output_text)
                solve_parsed.append(parsed)
                solve_correct.append(int(parsed == example.gold_text) if parsed is not None else 0)
                provisional_answers.append(parsed if parsed is not None else extract_final_answer_segment(output_text) or "UNPARSEABLE")

            bind_prompts = [
                build_binding_prompt(
                    tokenizer=scorer.tokenizer,
                    system_prompt=system_prompt,
                    cue_text=cue_payloads[example.item_id][0],
                    cue_placement=cue_placement,
                    formatting_style=formatting_style,
                    question_with_options=option_packs[example.item_id].question_with_options,
                    provisional_answer=provisional_answer,
                )
                for example, provisional_answer in zip(examples, provisional_answers, strict=True)
            ]
            bind_outputs = scorer.generate_batch(
                bind_prompts,
                max_new_tokens=int(max_tokens["binding_only"]),
                batch_size=resolve_batch_size(batch_size_config, "binding_only"),
            )
            rows = rows_for_examples(
                examples=examples,
                condition=condition,
                cue_payloads=cue_payloads,
                option_packs=option_packs,
                model_alias=model_payload["alias"],
                model_id=model_payload["model_id"],
                eval_mode="cot_before_options",
                cue_placement=cue_placement,
                formatting_style=formatting_style,
                seed=seed,
                config_digest=config_digest,
            )
            for row, example, solve_prompt, solve_output, parsed_answer, provisional_answer, solve_ok, bind_prompt, bind_output in zip(
                rows,
                examples,
                solve_prompts,
                solve_outputs,
                solve_parsed,
                provisional_answers,
                solve_correct,
                bind_prompts,
                bind_outputs,
                strict=True,
            ):
                option_pack = option_packs[example.item_id]
                bind_letter = parse_letter_answer(bind_output, set(option_pack.option_mapping))
                bind_ok = int(bind_letter == option_pack.gold_letter) if bind_letter is not None else 0
                row.update(
                    {
                        "solve_prompt_text": solve_prompt,
                        "binding_prompt_text": bind_prompt,
                        "solve_output_text": solve_output,
                        "binding_output_text": bind_output,
                        "provisional_answer": provisional_answer,
                        "parsed_solve_answer": parsed_answer or "",
                        "parsed_binding_letter": bind_letter or "",
                        "solve_correct": solve_ok,
                        "binding_correct": bind_ok,
                        "final_correct": int(bool(solve_ok and bind_ok)),
                        "parse_failure_solve": int(parsed_answer is None),
                        "parse_failure_bind": int(bind_letter is None),
                    }
                )
            all_rows.extend(rows)

        if "standard_mc" in config["evaluation"]["evaluation_modes"]:
            prompts = [
                build_standard_mc_prompt(
                    tokenizer=scorer.tokenizer,
                    system_prompt=system_prompt,
                    cue_text=cue_payloads[example.item_id][0],
                    cue_placement=cue_placement,
                    formatting_style=formatting_style,
                    question_with_options=option_packs[example.item_id].question_with_options,
                )
                for example in examples
            ]
            outputs = scorer.generate_batch(
                prompts,
                max_new_tokens=int(max_tokens["standard_mc"]),
                batch_size=resolve_batch_size(batch_size_config, "standard_mc"),
            )
            rows = rows_for_examples(
                examples=examples,
                condition=condition,
                cue_payloads=cue_payloads,
                option_packs=option_packs,
                model_alias=model_payload["alias"],
                model_id=model_payload["model_id"],
                eval_mode="standard_mc",
                cue_placement=cue_placement,
                formatting_style=formatting_style,
                seed=seed,
                config_digest=config_digest,
            )
            for row, example, prompt_text, output_text in zip(rows, examples, prompts, outputs, strict=True):
                option_pack = option_packs[example.item_id]
                parsed = parse_letter_answer(output_text, set(option_pack.option_mapping))
                correct = int(parsed == option_pack.gold_letter) if parsed is not None else 0
                row.update(
                    {
                        "solve_prompt_text": prompt_text,
                        "binding_prompt_text": "",
                        "solve_output_text": output_text,
                        "binding_output_text": "",
                        "provisional_answer": "",
                        "parsed_solve_answer": "",
                        "parsed_binding_letter": parsed or "",
                        "solve_correct": pd.NA,
                        "binding_correct": correct,
                        "final_correct": correct,
                        "parse_failure_solve": pd.NA,
                        "parse_failure_bind": int(parsed is None),
                    }
                )
            all_rows.extend(rows)

        if "binding_only" in config["evaluation"]["evaluation_modes"]:
            prompts = [
                build_binding_prompt(
                    tokenizer=scorer.tokenizer,
                    system_prompt=system_prompt,
                    cue_text=cue_payloads[example.item_id][0],
                    cue_placement=cue_placement,
                    formatting_style=formatting_style,
                    question_with_options=option_packs[example.item_id].question_with_options,
                    provisional_answer=example.gold_text,
                )
                for example in examples
            ]
            outputs = scorer.generate_batch(
                prompts,
                max_new_tokens=int(max_tokens["binding_only"]),
                batch_size=resolve_batch_size(batch_size_config, "binding_only"),
            )
            rows = rows_for_examples(
                examples=examples,
                condition=condition,
                cue_payloads=cue_payloads,
                option_packs=option_packs,
                model_alias=model_payload["alias"],
                model_id=model_payload["model_id"],
                eval_mode="binding_only",
                cue_placement=cue_placement,
                formatting_style=formatting_style,
                seed=seed,
                config_digest=config_digest,
            )
            for row, example, prompt_text, output_text in zip(rows, examples, prompts, outputs, strict=True):
                option_pack = option_packs[example.item_id]
                parsed = parse_letter_answer(output_text, set(option_pack.option_mapping))
                correct = int(parsed == option_pack.gold_letter) if parsed is not None else 0
                row.update(
                    {
                        "solve_prompt_text": "",
                        "binding_prompt_text": prompt_text,
                        "solve_output_text": "",
                        "binding_output_text": output_text,
                        "provisional_answer": example.gold_text,
                        "parsed_solve_answer": example.gold_text,
                        "parsed_binding_letter": parsed or "",
                        "solve_correct": 1,
                        "binding_correct": correct,
                        "final_correct": correct,
                        "parse_failure_solve": 0,
                        "parse_failure_bind": int(parsed is None),
                    }
                )
            all_rows.extend(rows)

        flush_outputs()
        elapsed = time.time() - condition_start
        print(
            f"[condition-done] model={model_payload['alias']} synthetic={config['data']['synthetic_path']} "
            f"condition={condition} rows={len(all_rows)} elapsed_sec={elapsed:.1f}",
            flush=True,
        )
    flush_outputs()


if __name__ == "__main__":
    main()
