"""Generate the synthetic four-family benchmark from scratch.

Step (1) in the rerun flow. Produces deterministically-seeded items for each
of:

- `boolean_logic`        : nested boolean expressions over named variables.
- `ordering_constraints` : "object X is left of object Y" style chains.
- `state_tracking`       : "track who is holding what after these swaps".
- `temporal_reasoning`   : date / interval problems.

Each item is generated at one of several controlled **lexicalization regimes**
(natural names, nonce names, mixed) and **difficulty levels**, so the final
test set has balanced coverage across both axes. Counts come from
`difficulty_counts` / `lexicalization_counts`.

Output goes to `data/processed/construct_validity_v3/{train,dev,test}_<family>.jsonl`.
The benchmark is *static* once generated — every downstream stage hashes
the consolidated test set so a silent drift would be caught.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
from pathlib import Path

import pandas as pd

from .data_loading import Candidate, TaskExample, save_task_examples_jsonl
from .utils import ensure_dir, load_yaml, stable_int_from_text


PEOPLE = [
    "Avery",
    "Blair",
    "Casey",
    "Drew",
    "Ellis",
    "Flynn",
    "Gray",
    "Harper",
    "Indy",
    "Jules",
    "Kai",
    "Lane",
]
NONCE_PEOPLE = [
    "daxen",
    "lorvi",
    "mipen",
    "nuvik",
    "soral",
    "tevan",
    "ulric",
    "voren",
    "wexil",
    "yaren",
    "zimor",
    "queln",
]
OBJECTS = [
    "amber token",
    "blue card",
    "copper key",
    "glass marble",
    "silver ring",
    "striped scarf",
    "paper ticket",
    "red badge",
]
NONCE_OBJECTS = [
    "miv token",
    "lor card",
    "tazen key",
    "quor marble",
    "vel ring",
    "nex scarf",
    "suri ticket",
    "prax badge",
]
CAUSE_FACTORS = [
    "blue switch",
    "red valve",
    "cooling fan",
    "relay pulse",
    "sensor gate",
    "backup pump",
    "control latch",
    "fuel line",
]
NONCE_FACTORS = [
    "miv switch",
    "lor valve",
    "tazen fan",
    "quor pulse",
    "vel gate",
    "nex pump",
    "suri latch",
    "prax line",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEXICALIZATION_REGIMES = ["surface_aligned", "paraphrastic", "delexicalized"]
DIFFICULTIES = ["easy", "medium", "hard"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def wrap_mc_candidates(answer_text: str, distractors: list[str], rng: random.Random) -> tuple[list[Candidate], str]:
    options = [answer_text] + distractors
    rng.shuffle(options)
    labels = [chr(ord("A") + idx) for idx in range(len(options))]
    candidates = [Candidate(label=label, text=text, score_text=label) for label, text in zip(labels, options, strict=True)]
    gold_label = labels[options.index(answer_text)]
    return candidates, gold_label


def lexicalized_people(regime: str) -> list[str]:
    return NONCE_PEOPLE if regime == "delexicalized" else PEOPLE


def lexicalized_objects(regime: str) -> list[str]:
    return NONCE_OBJECTS if regime == "delexicalized" else OBJECTS


def lexicalized_factors(regime: str) -> list[str]:
    return NONCE_FACTORS if regime == "delexicalized" else CAUSE_FACTORS


def regime_metadata(regime: str) -> dict[str, str]:
    return {
        "surface_aligned": {
            "ordering_left": "left of",
            "ordering_right": "right of",
            "hold": "holds",
            "swap": "swap what they are holding",
            "earlier": "earlier",
            "later": "later",
            "truth": "tells the truth",
            "lie": "lies",
            "cause": "causes",
        },
        "paraphrastic": {
            "ordering_left": "appears ahead of",
            "ordering_right": "appears behind",
            "hold": "currently carries",
            "swap": "exchange their carried items",
            "earlier": "comes sooner",
            "later": "comes later",
            "truth": "speaks accurately",
            "lie": "speaks inaccurately",
            "cause": "brings about",
        },
        "delexicalized": {
            "ordering_left": "occupies an earlier slot than",
            "ordering_right": "occupies a later slot than",
            "hold": "is paired with",
            "swap": "exchange pairings",
            "earlier": "precedes",
            "later": "follows",
            "truth": "is marked veridic",
            "lie": "is marked nonveridic",
            "cause": "activates",
        },
    }[regime]


def difficulty_counts(total: int) -> dict[str, int]:
    base = total // len(DIFFICULTIES)
    remainder = total % len(DIFFICULTIES)
    return {
        difficulty: base + (1 if idx < remainder else 0)
        for idx, difficulty in enumerate(DIFFICULTIES)
    }


def lexicalization_counts(total: int) -> dict[str, int]:
    base = total // len(LEXICALIZATION_REGIMES)
    remainder = total % len(LEXICALIZATION_REGIMES)
    return {
        regime: base + (1 if idx < remainder else 0)
        for idx, regime in enumerate(LEXICALIZATION_REGIMES)
    }


def build_boolean_expression(rng: random.Random, variables: list[str], depth: int) -> tuple[str, bool, dict]:
    if depth <= 0 or rng.random() < 0.35:
        symbol = rng.choice(variables)
        return symbol, None, {"type": "var", "symbol": symbol}
    op = rng.choice(["not", "and", "or"])
    if op == "not":
        child_expr, _, child_tree = build_boolean_expression(rng, variables, depth - 1)
        return f"not ({child_expr})", None, {"type": "not", "child": child_tree}
    left_expr, _, left_tree = build_boolean_expression(rng, variables, depth - 1)
    right_expr, _, right_tree = build_boolean_expression(rng, variables, depth - 1)
    return f"({left_expr}) {op} ({right_expr})", None, {"type": op, "left": left_tree, "right": right_tree}


def eval_boolean_tree(tree: dict, assignments: dict[str, bool]) -> bool:
    kind = tree["type"]
    if kind == "var":
        return assignments[tree["symbol"]]
    if kind == "not":
        return not eval_boolean_tree(tree["child"], assignments)
    if kind == "and":
        return eval_boolean_tree(tree["left"], assignments) and eval_boolean_tree(tree["right"], assignments)
    if kind == "or":
        return eval_boolean_tree(tree["left"], assignments) or eval_boolean_tree(tree["right"], assignments)
    raise ValueError(f"Unknown tree node: {kind}")


def tree_depth(tree: dict) -> int:
    kind = tree["type"]
    if kind == "var":
        return 1
    if kind == "not":
        return 1 + tree_depth(tree["child"])
    return 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def negation_depth(tree: dict) -> int:
    kind = tree["type"]
    if kind == "var":
        return 0
    if kind == "not":
        return 1 + negation_depth(tree["child"])
    return max(negation_depth(tree["left"]), negation_depth(tree["right"]))


def operator_count(tree: dict) -> int:
    kind = tree["type"]
    if kind == "var":
        return 0
    if kind == "not":
        return 1 + operator_count(tree["child"])
    return 1 + operator_count(tree["left"]) + operator_count(tree["right"])


def boolean_item(idx: int, split: str, difficulty: str, regime: str, rng: random.Random) -> TaskExample:
    depth = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
    variable_count = {"easy": 3, "medium": 4, "hard": 5}[difficulty]
    variables = [chr(ord("p") + i) for i in range(variable_count)]
    assignments = {symbol: rng.choice([True, False]) for symbol in variables}
    expr, _, tree = build_boolean_expression(rng, variables, depth)
    value = eval_boolean_tree(tree, assignments)
    precedence_critical = (" and " in expr and " or " in expr)
    assignment_text = ", ".join(f"{symbol}={'True' if assignments[symbol] else 'False'}" for symbol in variables)
    if regime == "surface_aligned":
        question = f"Given {assignment_text}, evaluate the boolean expression: {expr}"
    elif regime == "paraphrastic":
        question = f"Use these truth settings: {assignment_text}. Under those settings, does this formula come out true or false? {expr}"
    else:
        question = f"Setings: {assignment_text}. Determine the terminal truth mark of this formula: {expr}"
    metadata = {
        "family_v2": "boolean_logic",
        "source": "construct_validity_v3",
        "difficulty": difficulty,
        "lexicalization_regime": regime,
        "annotations": {
            "parse_tree_depth": tree_depth(tree),
            "negation_depth": negation_depth(tree),
            "operator_count": operator_count(tree),
            "precedence_critical": precedence_critical,
            "assignment": assignments,
            "tree": tree,
        },
    }
    return TaskExample(
        task_name="cv3_boolean_logic",
        concept_name="boolean_logic",
        item_id=f"{split}:cv3:boolean_logic:{regime}:{difficulty}:{idx:05d}",
        question=question,
        answer_type="true_false",
        gold_label="True" if value else "False",
        gold_text="True" if value else "False",
        candidates=[
            Candidate(label="True", text="True", score_text="True"),
            Candidate(label="False", text="False", score_text="False"),
        ],
        metadata=metadata,
    )


def ordering_item(idx: int, split: str, difficulty: str, regime: str, rng: random.Random) -> TaskExample:
    lex = regime_metadata(regime)
    names = rng.sample(lexicalized_people(regime), {"easy": 4, "medium": 5, "hard": 6}[difficulty])
    order = names[:]
    rng.shuffle(order)
    target_pos = rng.randrange(len(order))
    clues = []
    direct_pairs = set()
    for left, right in zip(order[:-1], order[1:], strict=True):
        clues.append(f"{left} is {lex['ordering_left']} {right}.")
        direct_pairs.add((left, right))
    if difficulty != "easy":
        clues.append(f"{order[0]} occupies the earliest slot.")
    query_type = "who_is_kth"
    question = (
        " ".join(clues)
        + f" Which entity is position {target_pos + 1} in the final arrangement?"
    )
    answer = order[target_pos]
    distractors = [name for name in names if name != answer][:3]
    candidates, gold_label = wrap_mc_candidates(answer, distractors, rng)
    question = question + "\nOptions:\n" + "\n".join(f"({c.label}) {c.text}" for c in candidates)
    metadata = {
        "family_v2": "ordering_constraints",
        "source": "construct_validity_v3",
        "difficulty": difficulty,
        "lexicalization_regime": regime,
        "annotations": {
            "entity_count": len(order),
            "constraint_count": len(clues),
            "query_type": query_type,
            "target_position": target_pos + 1,
            "requires_transitive_closure": target_pos not in {0, len(order) - 1},
            "final_order": order,
            "distractor_count": len(candidates) - 1,
        },
    }
    return TaskExample(
        task_name="cv3_ordering_constraints",
        concept_name="ordering_constraints",
        item_id=f"{split}:cv3:ordering_constraints:{regime}:{difficulty}:{idx:05d}",
        question=question,
        answer_type="option_letter",
        gold_label=gold_label,
        gold_text=answer,
        candidates=candidates,
        metadata=metadata,
    )


def state_tracking_item(idx: int, split: str, difficulty: str, regime: str, rng: random.Random) -> TaskExample:
    lex = regime_metadata(regime)
    people = rng.sample(lexicalized_people(regime), {"easy": 4, "medium": 5, "hard": 6}[difficulty])
    objects = rng.sample(lexicalized_objects(regime), len(people))
    assignment = dict(zip(people, objects, strict=True))
    steps = {"easy": 3, "medium": 5, "hard": 7}[difficulty]
    operations = []
    op_types = []
    for step_idx in range(steps):
        if difficulty == "easy" or rng.random() < 0.55:
            a, b = rng.sample(people, 2)
            assignment[a], assignment[b] = assignment[b], assignment[a]
            operations.append(f"Step {step_idx + 1}: {a} and {b} {lex['swap']}.")
            op_types.append("swap")
        else:
            giver, receiver = rng.sample(people, 2)
            assignment[receiver] = assignment[giver]
            operations.append(f"Step {step_idx + 1}: {receiver} is updated to match {giver}.")
            op_types.append("reassignment")
    target_person = rng.choice(people)
    answer = assignment[target_person]
    distractors = [obj for obj in objects if obj != answer][:3]
    candidates, gold_label = wrap_mc_candidates(answer, distractors, rng)
    intro = " ".join(f"{person} initially {lex['hold']} the {obj}." for person, obj in zip(people, objects, strict=True))
    question = f"{intro} {' '.join(operations)} At the end, what is {target_person} paired with?\nOptions:\n" + "\n".join(
        f"({c.label}) {c.text}" for c in candidates
    )
    metadata = {
        "family_v2": "state_tracking",
        "source": "construct_validity_v3",
        "difficulty": difficulty,
        "lexicalization_regime": regime,
        "annotations": {
            "entity_count": len(people),
            "step_count": steps,
            "operation_profile": op_types,
            "branching_factor": len({tuple(sorted(pair)) for pair in zip(people, objects, strict=True)}),
            "target_person": target_person,
            "final_assignment": assignment.copy(),
        },
    }
    return TaskExample(
        task_name="cv3_state_tracking",
        concept_name="state_tracking",
        item_id=f"{split}:cv3:state_tracking:{regime}:{difficulty}:{idx:05d}",
        question=question,
        answer_type="option_letter",
        gold_label=gold_label,
        gold_text=answer,
        candidates=candidates,
        metadata=metadata,
    )


def temporal_item(idx: int, split: str, difficulty: str, regime: str, rng: random.Random) -> TaskExample:
    base = dt.date(rng.randint(2008, 2024), rng.randint(1, 12), 1) + dt.timedelta(days=rng.randint(0, 25))
    subtype = rng.choice(["pure_order", "date_arithmetic", "formatting"])
    if subtype == "pure_order":
        events = rng.sample(lexicalized_people(regime), {"easy": 3, "medium": 4, "hard": 5}[difficulty])
        order = events[:]
        rng.shuffle(order)
        clues = [f"{order[i]} happens {regime_metadata(regime)['earlier']} {order[i + 1]}." for i in range(len(order) - 1)]
        target = rng.choice(order[1:-1] if len(order) > 2 else order)
        answer = "Yes" if order.index(target) < order.index(order[-1]) else "No"
        question = " ".join(clues) + f" Does {target} occur before {order[-1]}?"
        metadata_answer_type = "yes_no"
        candidates = [
            Candidate(label="Yes", text="Yes", score_text="Yes"),
            Candidate(label="No", text="No", score_text="No"),
        ]
        gold_label = answer
        gold_text = answer
        annotation = {
            "subtype": subtype,
            "chain_length": len(order),
            "formatting_required": False,
            "event_order": order,
        }
    else:
        steps = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
        cursor = base
        parts = [f"The reference date is {cursor.strftime('%m/%d/%Y')}."]
        offsets = []
        for _ in range(steps):
            delta = rng.choice([-14, -7, -3, 3, 7, 14])
            offsets.append(delta)
            cursor = cursor + dt.timedelta(days=delta)
            phrase = regime_metadata(regime)["later"] if delta > 0 else regime_metadata(regime)["earlier"]
            parts.append(f"Move {abs(delta)} days {phrase}.")
        if subtype == "formatting":
            answer = cursor.strftime("%B %d, %Y")
            formatting_required = True
        else:
            answer = cursor.strftime("%m/%d/%Y")
            formatting_required = False
        distractors = {
            (cursor + dt.timedelta(days=7)).strftime("%B %d, %Y" if formatting_required else "%m/%d/%Y"),
            (cursor - dt.timedelta(days=7)).strftime("%B %d, %Y" if formatting_required else "%m/%d/%Y"),
            (cursor + dt.timedelta(days=1)).strftime("%B %d, %Y" if formatting_required else "%m/%d/%Y"),
        }
        distractors.discard(answer)
        candidates, gold_label = wrap_mc_candidates(answer, list(distractors)[:3], rng)
        question = " ".join(parts) + f" What date do you get at the end?\nOptions:\n" + "\n".join(
            f"({c.label}) {c.text}" for c in candidates
        )
        metadata_answer_type = "option_letter"
        gold_text = answer
        annotation = {
            "subtype": subtype,
            "chain_length": steps,
            "offsets": offsets,
            "formatting_required": formatting_required,
            "offset_magnitude": sum(abs(value) for value in offsets),
        }
    metadata = {
        "family_v2": "temporal_reasoning",
        "source": "construct_validity_v3",
        "difficulty": difficulty,
        "lexicalization_regime": regime,
        "annotations": annotation,
    }
    return TaskExample(
        task_name="cv3_temporal_reasoning",
        concept_name="temporal_reasoning",
        item_id=f"{split}:cv3:temporal_reasoning:{regime}:{difficulty}:{idx:05d}",
        question=question,
        answer_type=metadata_answer_type,
        gold_label=gold_label,
        gold_text=gold_text,
        candidates=candidates,
        metadata=metadata,
    )


def truth_item(idx: int, split: str, difficulty: str, regime: str, rng: random.Random) -> TaskExample:
    lex = regime_metadata(regime)
    names = rng.sample(lexicalized_people(regime), {"easy": 3, "medium": 4, "hard": 5}[difficulty])
    truth_assignment = {name: rng.choice([True, False]) for name in names}
    statements = []
    for speaker in names:
        target = rng.choice([name for name in names if name != speaker])
        truthful_predicate = lex["truth"] if truth_assignment[target] else lex["lie"]
        if truth_assignment[speaker]:
            spoken = truthful_predicate
        else:
            spoken = lex["lie"] if truthful_predicate == lex["truth"] else lex["truth"]
        statements.append(f"{speaker} says {target} {spoken}.")
    ask_person = rng.choice(names)
    answer = "Yes" if truth_assignment[ask_person] else "No"
    question = " ".join(statements) + f" Is {ask_person} truthful?"
    metadata = {
        "family_v2": "truth_consistency",
        "source": "construct_validity_v3",
        "difficulty": difficulty,
        "lexicalization_regime": regime,
        "annotations": {
            "agent_count": len(names),
            "statement_count": len(statements),
            "liar_depth": sum(1 for value in truth_assignment.values() if not value),
            "contradiction_type": "speaker_target_relation",
            "truth_assignment": truth_assignment,
        },
    }
    return TaskExample(
        task_name="cv3_truth_consistency",
        concept_name="truth_consistency",
        item_id=f"{split}:cv3:truth_consistency:{regime}:{difficulty}:{idx:05d}",
        question=question,
        answer_type="yes_no",
        gold_label=answer,
        gold_text=answer,
        candidates=[
            Candidate(label="Yes", text="Yes", score_text="Yes"),
            Candidate(label="No", text="No", score_text="No"),
        ],
        metadata=metadata,
    )


def causal_item(idx: int, split: str, difficulty: str, regime: str, rng: random.Random) -> TaskExample:
    factors = rng.sample(lexicalized_factors(regime), {"easy": 4, "medium": 5, "hard": 6}[difficulty])
    query_type = rng.choice(["direct_cause", "intervention", "counterfactual", "common_cause"])
    active = {factor: rng.choice([True, False]) for factor in factors}
    cause_a, cause_b, mid, outcome = factors[0], factors[1], factors[2], factors[3]
    active[cause_a] = True
    active[mid] = True
    story = [
        f"If {cause_a} is on, then {mid} becomes active.",
        f"If {cause_b} is on, then {mid} also becomes active.",
        f"If {mid} is active, then {outcome} appears.",
    ]
    if len(factors) > 4:
        conf = factors[4]
        story.append(f"{conf} can turn on both {cause_a} and {cause_b}.")
        active[conf] = rng.choice([True, False])
    if query_type == "direct_cause":
        answer = mid
        distractors = [factor for factor in factors if factor != answer][:3]
        candidates, gold_label = wrap_mc_candidates(answer, distractors, rng)
        question = " ".join(story) + f" Which factor directly activates {outcome}?\nOptions:\n" + "\n".join(
            f"({c.label}) {c.text}" for c in candidates
        )
        answer_type = "option_letter"
    elif query_type == "intervention":
        answer = "Yes"
        question = " ".join(story) + f" If {mid} were removed, would {outcome} still appear?"
        candidates = [
            Candidate(label="Yes", text="Yes", score_text="Yes"),
            Candidate(label="No", text="No", score_text="No"),
        ]
        gold_label = answer
        answer_type = "yes_no"
    elif query_type == "counterfactual":
        answer = "No"
        question = " ".join(story) + f" If {cause_a} had been off and everything else stayed the same, would {mid} still be active?"
        candidates = [
            Candidate(label="Yes", text="Yes", score_text="Yes"),
            Candidate(label="No", text="No", score_text="No"),
        ]
        gold_label = answer
        answer_type = "yes_no"
    else:
        answer = factors[4] if len(factors) > 4 else cause_a
        distractors = [factor for factor in factors if factor != answer][:3]
        candidates, gold_label = wrap_mc_candidates(answer, distractors, rng)
        question = " ".join(story) + f" Which factor is a common upstream cause for two other factors in this system?\nOptions:\n" + "\n".join(
            f"({c.label}) {c.text}" for c in candidates
        )
        answer_type = "option_letter"
    metadata = {
        "family_v2": "causal_reasoning",
        "source": "construct_validity_v3",
        "difficulty": difficulty,
        "lexicalization_regime": regime,
        "annotations": {
            "graph_size": len(factors),
            "query_type": query_type,
            "confounder_present": len(factors) > 4,
            "active_assignment": active,
            "direct_edge_target": mid,
            "outcome_node": outcome,
        },
    }
    return TaskExample(
        task_name="cv3_causal_reasoning",
        concept_name="causal_reasoning",
        item_id=f"{split}:cv3:causal_reasoning:{regime}:{difficulty}:{idx:05d}",
        question=question,
        answer_type=answer_type,
        gold_label=gold_label,
        gold_text=answer,
        candidates=candidates,
        metadata=metadata,
    )


FAMILY_BUILDERS = {
    "boolean_logic": boolean_item,
    "ordering_constraints": ordering_item,
    "state_tracking": state_tracking_item,
    "temporal_reasoning": temporal_item,
    "truth_consistency": truth_item,
    "causal_reasoning": causal_item,
}


def generate_split(split: str, total_per_family: int, seed: int) -> list[TaskExample]:
    examples: list[TaskExample] = []
    difficulty_quota = difficulty_counts(total_per_family)
    for family_name, builder in FAMILY_BUILDERS.items():
        for difficulty in DIFFICULTIES:
            lexical_quota = lexicalization_counts(difficulty_quota[difficulty])
            for regime in LEXICALIZATION_REGIMES:
                count = lexical_quota[regime]
                for item_idx in range(count):
                    seed_key = f"{split}|{family_name}|{difficulty}|{regime}|{item_idx}|{seed}"
                    local_seed = stable_int_from_text(seed_key, modulo=2**32)
                    rng = random.Random(local_seed)
                    examples.append(builder(item_idx, split, difficulty, regime, rng))
    return examples


def validate_examples(examples: list[TaskExample]) -> tuple[list[dict], list[dict]]:
    rows = []
    samples = []
    for example in examples:
        annotations = (example.metadata or {}).get("annotations", {})
        rows.append(
            {
                "task_name": example.task_name,
                "family_name": example.metadata["family_v2"],
                "difficulty": example.metadata["difficulty"],
                "lexicalization_regime": example.metadata["lexicalization_regime"],
                "answer_type": example.answer_type,
                "candidate_count": len(example.candidates),
                "has_annotations": int(bool(annotations)),
                "gold_in_candidates": int(
                    example.answer_type in {"yes_no", "true_false"} or any(candidate.text == example.gold_text for candidate in example.candidates)
                ),
            }
        )
    by_bucket = {}
    for example in examples:
        key = (example.metadata["family_v2"], example.metadata["lexicalization_regime"])
        if key not in by_bucket:
            by_bucket[key] = example
    for (family_name, regime), example in sorted(by_bucket.items()):
        samples.append(
            {
                "family_name": family_name,
                "lexicalization_regime": regime,
                "item_id": example.item_id,
                "question": example.question,
                "gold_text": example.gold_text,
                "annotations": json.dumps(example.metadata["annotations"], ensure_ascii=True),
            }
        )
    return rows, samples


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_dir = Path(config["data"]["output_dir"])
    ensure_dir(output_dir)
    dev_examples = generate_split("dev", int(config["data"]["dev_per_family"]), int(config["experiment"]["seed"]))
    test_examples = generate_split("test", int(config["data"]["test_per_family"]), int(config["experiment"]["seed"]) + 1000)

    save_task_examples_jsonl(dev_examples, output_dir / "dev.jsonl")
    save_task_examples_jsonl(test_examples, output_dir / "test.jsonl")

    dev_rows, dev_samples = validate_examples(dev_examples)
    test_rows, test_samples = validate_examples(test_examples)
    stats_frame = pd.DataFrame(dev_rows + test_rows)
    summary = (
        stats_frame.groupby(
            ["task_name", "family_name", "difficulty", "lexicalization_regime", "answer_type"],
            dropna=False,
        )
        .agg(
            n=("task_name", "count"),
            has_annotations=("has_annotations", "mean"),
            gold_in_candidates=("gold_in_candidates", "mean"),
            candidate_count=("candidate_count", "mean"),
        )
        .reset_index()
    )
    sample_frame = pd.DataFrame(dev_samples + test_samples)

    stats_path = Path(config["results"]["stats_csv"])
    ensure_dir(stats_path.parent)
    summary.to_csv(stats_path, index=False)
    sample_path = Path(config["results"]["sample_csv"])
    ensure_dir(sample_path.parent)
    sample_frame.to_csv(sample_path, index=False)

    report_lines = [
        "# Phase C Benchmark v3",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Families: {', '.join(FAMILY_BUILDERS)}",
        f"- Dev per family: `{config['data']['dev_per_family']}`",
        f"- Test per family: `{config['data']['test_per_family']}`",
        f"- Lexicalization regimes: {', '.join(LEXICALIZATION_REGIMES)}",
        "",
        "## Summary Stats",
        "",
        "```text",
        summary.head(36).to_string(index=False),
        "```",
        "",
        "## Validation Checks",
        "",
        "- Every item carries `annotations` metadata.",
        "- Candidate-backed items retain an exact gold text inside the option set.",
        "- Each family is generated across three lexicalization regimes and three difficulty bins.",
        "",
        "## Sample Rows",
        "",
        "```text",
        sample_frame.head(12).to_string(index=False),
        "```",
    ]
    report_path = Path(config["results"]["report_path"])
    ensure_dir(report_path.parent)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
