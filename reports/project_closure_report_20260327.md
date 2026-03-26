# Project Closure Report

## What This Project Was

This project studied whether very small prompt cues can steer large language model reasoning in a selective and concept-local way.

The original positive hypothesis was stronger than “cues matter.” It claimed that multi-variant semantic bundles, meaning several short paraphrastic cues tied to the same reasoning concept, would outperform simpler controls such as exact repetition, generic neutral text, canonical wording, lexical overlap, or procedural hints, and would selectively steer reasoning toward the matched reasoning family.

By the end of the project, that broad claim was not supported.

## The Main Research Question At Closure

The project ended with a narrower and more defensible question:

“When answer binding is separated from solving, do minimal cue effects in LLM reasoning live mainly in canonical/operator/lexical/interface effects rather than paraphrastic semantic flooding?”

In plain language:

- Do micro-cues really change reasoning because of their semantic meaning?
- Or do they mostly work because they change operator setup, wording overlap, canonical phrasing, or prompt placement?

## How The Project Evolved

The project went through several stages.

1. Early broad evaluations:
   Natural and synthetic evaluations suggested that cues could change behavior, but the evidence was not clean enough to separate true reasoning changes from answer-format or interface effects.

2. Larger audits:
   Natural audits and synthetic subset diagnostics weakened the original broad semantic-flooding story. In those analyses, matched semantic bundles did not reliably beat all main controls. Single-canonical cues were often stronger.

3. Construct-valid synthetic benchmark work:
   The project then shifted toward a cleaner synthetic benchmark with controllable families, difficulty, and lexicalization regimes. This led to benchmark v3, which emphasized construct validity over broad coverage.

4. Decoupled evaluation:
   A key methodological move was to separate solving from answer binding. This let the project ask whether a cue improves reasoning itself or only helps the model map a solution into a final answer format.

5. Phase E held-out confirmatory:
   A held-out synthetic confirmatory on Qwen2.5-7B showed that semantic flooding remained weak under decoupled evaluation. Binding-only was near ceiling, which implied the fragile part was the solve stage, not answer binding.

6. Phase R0 rigour audit:
   Because the Phase E work was done across interrupted sessions, a provenance audit was performed. The old Phase E results were found to be directionally useful but not publication-grade, so a clean confirmatory namespace was created.

7. Phase R1 multimodel confirmatory:
   The final strong evidence in this repository comes from the clean four-model confirmatory rerun on the frozen held-out benchmark slice.

## Final Experimental Setup

The final clean experiment preserved in this archive has these components.

- Benchmark family source: construct-validity benchmark v3.
- Full benchmark v3 families: boolean logic, ordering constraints, state tracking, temporal reasoning, truth consistency, causal reasoning.
- Frozen confirmatory slice: four families only.
  Those four were `boolean_logic`, `ordering_constraints`, `state_tracking`, and `temporal_reasoning`.
- Frozen confirmatory size: 7,200 items total.
- Lexicalization regimes in the benchmark generator: `surface_aligned`, `paraphrastic`, `delexicalized`.
- Evaluation modes:
  `free_form_only`, `cot_before_options`, `standard_mc`, `binding_only`.
- Paper-facing primary modes:
  `free_form_only` and `cot_before_options`.
- Cue conditions in the final confirmatory:
  `no_cue`, `generic_neutral`, `single_canonical`, `exact_repetition`, `matched_semantic`, `matched_procedural`, `matched_mixed`, `matched_lexical_overlap`, `matched_near_miss`.
- Frozen prompt interface:
  cue placement `after_question`, formatting `plain_sentence`.

Final clean model registry:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`
- `mistralai/Ministral-8B-Instruct-2410`

Notes on model access:

- `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-2-9b-it` were planned as possible replication targets.
- In this environment, those gated weights were not available at download time.
- `Ministral-8B-Instruct-2410` was therefore used as the open fallback model.

## What Was Learned

### The Broad Original Claim Failed

The repository does not support the claim that paraphrastic same-concept semantic bundles are the dominant or uniquely robust source of cue gains.

Under the clean multimodel confirmatory:

- All four registry runs completed.
- All four passed completion and integrity checks.
- In the pooled primary-mode analysis, `matched_semantic` was worse than `no_cue`.
- It was also worse than `single_canonical`.
- It was also worse than `matched_lexical_overlap`.
- It was also worse than `exact_repetition`.

The only pooled contrast where `matched_semantic` came out positive was against `matched_procedural`, and even there the effect was small and mixed across models.

### The Solve Stage Was The Vulnerable Part

One of the most important lessons from the project is that answer binding and problem solving must be separated.

- `binding_only` was near ceiling.
- The instability appeared mainly in the solve stage.
- This means many apparent cue gains can be inflated or distorted if solving and answer binding are not decoupled.

### What Seems Real

The project does support a narrower claim:

- Minimal cues can change behavior.
- But the stable residue is not broad semantic flooding.
- The remaining signal is better described as a mixture of:
  canonical phrasing,
  lexical overlap,
  procedural/operator hints,
  and prompt-interface effects.

That does not mean every such effect is strong.
It means those factors were consistently more plausible than the broad semantic-flooding story.

## Clean Final Findings To Carry Forward

If someone only remembers three conclusions from this project, they should be these:

1. Broad semantic flooding was not replicated under construct-valid, decoupled evaluation.
2. Answer binding can hide or inflate apparent gains, so decoupled evaluation is necessary.
3. The most defensible remaining story is a decomposition or methods story, not a broad positive semantic-steering story.

## What Was Not Resolved

The project did not fully settle every narrower question.

- Procedural/operator residue was mixed rather than cleanly dominant.
- Placement and interface effects were known to matter a lot, but the final placement-factorial branch was not completed into a full paper package.
- Operator-state probing was not completed into a strong mechanistic result.

So the project ended with a strong falsification / decomposition result, but not with a complete mechanistic account.

## Publication Read At Closure

The most defensible paper direction at closure would have been one of these:

- A benchmark-and-decomposition paper.
- A methods/falsification paper about decoupled evaluation and interface-sensitive cue effects.

The least defensible direction would be to revive the original positive semantic-flooding framing.

If this project is ever revived, it should not return to the old broad claim.

## Status Of The Final Clean Runs

The final clean multimodel confirmatory is complete.

- All four models completed `259,200` rows each.
- `run_manifest` reported every model as `completed`.
- `integrity_check` reported no duplicate keys, no missing keys, no extra keys, and consistent cross-model interface fields.

Those results were summarized in the final confirmatory report during closure.

## What This Archive Intentionally Preserves

This archive is intentionally small.

It preserves the minimal code needed to understand and rerun the final, clean branch of the project:

- benchmark generation
- frozen confirmatory input preparation
- decoupled evaluation execution
- confirmatory postprocessing
- prompt assets
- final configs
- this closure report

It intentionally does **not** preserve the large raw results directories.

That was a deliberate choice because this archive is meant for long-term storage of the essential runnable code and the final interpretation, not for keeping every heavy intermediate artifact.

## Included Files And Why They Matter

Included source files:

- `src/__init__.py`
- `src/utils.py`
- `src/data_loading.py`
- `src/prompt_building.py`
- `src/scoring_methods.py`
- `src/score_mc.py`
- `src/stats.py`
- `src/build_benchmark_v3.py`
- `src/prepare_confirmatory_v3_inputs.py`
- `src/run_decoupled_eval.py`
- `src/postprocess_confirmatory_v3.py`

Included configs:

- `configs/construct_validity_phaseC_v3.yaml`
- `configs/confirmatory_v3_qwen7b.yaml`
- `configs/confirmatory_v3_qwen14b.yaml`
- `configs/confirmatory_v3_deepseek7b.yaml`
- `configs/confirmatory_v3_ministral8b.yaml`

Included prompt assets:

- `prompts/bundles_v3.yaml`
- `prompts/templates_v2.yaml`

Included environment spec:

- `requirements.txt`

Included report:

- `reports/project_closure_report_20260327.md`

## How To Recreate The Final Pipeline

From the archive root, the core workflow is:

1. Build the construct-valid benchmark:

```bash
python -m src.build_benchmark_v3 --config configs/construct_validity_phaseC_v3.yaml
```

2. Prepare the frozen four-family confirmatory slice:

```bash
python -m src.prepare_confirmatory_v3_inputs
```

3. Run the decoupled confirmatory for the desired model configs:

```bash
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen7b.yaml --gpu-id 0
python -m src.run_decoupled_eval --config configs/confirmatory_v3_qwen14b.yaml --gpu-id 1
python -m src.run_decoupled_eval --config configs/confirmatory_v3_deepseek7b.yaml --gpu-id 2
python -m src.run_decoupled_eval --config configs/confirmatory_v3_ministral8b.yaml --gpu-id 3
```

4. Postprocess the completed runs:

```bash
python -m src.postprocess_confirmatory_v3 --bootstrap-samples 1000
```

The `1000` bootstrap setting was used at closure to keep postprocessing tractable while still producing paired confidence intervals and verdict labels.

## Recommended Interpretation For Any Future Reader

The shortest honest summary is:

This was a research project about micro-cue steering in LLM reasoning. It began with a broad positive hypothesis about semantic flooding, but the strongest clean evidence did not support that story. The project’s most valuable output is instead a construct-valid benchmark and a decoupled evaluation pipeline showing that many apparent cue gains are better understood as canonical, lexical, procedural, or interface effects rather than paraphrastic semantic steering.

## Closure Recommendation

If the project stays closed, this archive is enough to preserve what mattered.

If the project is reopened in the future, start from the benchmark/protocol/decomposition framing.
Do not restart from the old semantic-flooding mainline.
