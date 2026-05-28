# Glossary

The closure reports and the code use a few internal terms that aren't
self-explanatory if you're coming in cold. This is the decoder ring.

## Concepts and families

| Term | What it means |
|---|---|
| **Concept** | One of six target reasoning concepts: `causality`, `temporal`, `ordering`, `state_tracking`, `boolean_logic`, `consistency_truth`. Each BBH task is tagged with one. See `TASK_TO_CONCEPT` in `src/data_loading.py`. |
| **Family** (a.k.a. v2 family) | The coarser six-bucket taxonomy used by the final synthetic benchmark: `boolean_logic`, `ordering_constraints`, `state_tracking`, `temporal_reasoning`, `truth_consistency`, `causal_reasoning`. `V1_TO_V2` in `src/scoring_methods.py` maps the old concept names to these. |
| **V1 vs V2** | The "concept" taxonomy is the legacy (v1) names that BBH tasks land in; the "family" taxonomy is the cleaner v2 grouping the final benchmark and reports use. The mapping is one-to-one in most cases (e.g., `temporal` → `temporal_reasoning`). |

## Conditions ("which cue was in the prompt")

| Condition | What got injected before the question |
|---|---|
| `no_cue` | Nothing — bare question. |
| `generic_neutral_bundle` | A fixed, content-free preamble (cue length control). |
| `exact_repetition` | The same canonical cue line repeated N times. |
| `single_canonical` | The canonical cue line, exactly once. |
| `matched_semantic` / `concept_bundle_*` | Several rephrasings of the canonical line that **match the question's concept**. The headline hypothesis. |
| `matched_procedural` | Procedural-flavored variants of the canonical (close to the concept but presented as a how-to). |
| `matched_lexical_overlap` | Variants that share words with the question but not its concept (lexical-overlap control). |
| `matched_mixed` / `matched_near_miss` | Secondary controls; partial overlaps. |

## Evaluation modes ("how the answer was extracted")

| Mode | What the model sees, what we record |
|---|---|
| `free_form_only` | Question only. Model generates; we parse a label out. |
| `cot_before_options` | Model generates a chain-of-thought, then we score options against the produced text. |
| `standard_mc` | Question with options. Score each option's log-probability; pick the highest. |
| `binding_only` | Question **with the gold answer text already written into the prompt**. Score each option label; if the model can pick the right letter, it isn't the "solve" step failing — it's the "binding" step. |

The pair `(standard_mc, binding_only)` is what the closure report calls the
**decoupled evaluation**: it lets us split "did the model solve the problem"
from "did it map the solution onto an answer choice."

## Phases and run names

| Term | What it refers to |
|---|---|
| `phaseC` | The construct-validity round — sanity-checking that the synthetic benchmark has the difficulty / lexicalization coverage it's supposed to. Produced under `data/processed/construct_validity_v3/`. |
| `phaseE_heldout_4family` | The held-out test split that the final confirmatory evaluation runs on. Four families: boolean_logic, ordering_constraints, state_tracking, temporal_reasoning. |
| `confirmatory_v3` | The final round that produced the headline numbers in the closure report. The `_v3` is the project's internal version counter; it's not meaningful outside the project. |
| `v1`, `v2`, `v3` suffixes | Internal iteration counters. `v3` is the last one and everything in this repo is the v3 state of the world. |

## Metrics

| Name | Definition |
|---|---|
| **Diagonal advantage** | `matched concept accuracy − off-diagonal mean accuracy`, per (model, task). Positive = the cue helps when the concept matches. |
| **Selective Steering Index (SSI)** | Same thing — the diagonal advantage. See `selective_steering_index` in `src/stats.py`. |
| **Matched-beats-controls** | Matched concept accuracy > best of {no_cue, generic_neutral_bundle, exact_repetition}, per task. |
| **Pilot go / no-go** | The pre-registered decision rule: needs ≥ 3 tasks where matched beats controls AND ≥ 4 tasks with positive diagonal advantage. Result was **no go**. |

## Why the version suffixes are still in filenames

The closure reports name artifacts by their on-disk paths
(`phaseE_heldout_4family_test.jsonl`, `bundles_v3.yaml`, ...). Renaming the
files would silently break those cross-references, so the filenames are
preserved as historical record. This glossary is the bridge.
