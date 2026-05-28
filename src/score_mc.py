"""Multiple-choice scoring on a Hugging Face causal LM.

Wraps a 4-bit quantized HF model behind `MultipleChoiceScorer.score_batch`,
which returns log-probabilities for each candidate completion given the same
prompt prefix. Used by `scoring_methods.score_prepared_examples`.

The constructor handles:

- Optional 4-bit nf4 quantization via `bitsandbytes`.
- Padding token shim (sets `pad_token = eos_token` if missing).
- Attention implementation override (`eager` for older GPUs, `sdpa` for
  newer ones).

`ScoredCandidate` carries both summed and length-normalized log-probs; the
selection rule lives in `scoring_methods.candidate_value`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ScoredCandidate:
    label: str
    text: str
    score_text: str
    logprob_sum: float
    token_count: int
    first_token_logprob: float
    length_norm_logprob: float


class MultipleChoiceScorer:
    def __init__(
        self,
        model_id: str,
        gpu_id: int,
        load_in_4bit: bool = True,
        compute_dtype: str = "bfloat16",
        attn_implementation: str | None = None,
    ) -> None:
        dtype = getattr(torch, compute_dtype)
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        model_kwargs: dict[str, Any] = {
            "device_map": {"": gpu_id},
            "torch_dtype": dtype,
        }
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if getattr(self.model.config, "use_sliding_window", False):
            self.model.config.use_sliding_window = False
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.do_sample = False
            generation_config.temperature = 1.0
            generation_config.top_p = 1.0
            generation_config.top_k = 50
        self.model.eval()

    def encode_prompt(self, prompt_text: str) -> list[int]:
        return self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    def encode_candidate(self, answer_text: str) -> list[int]:
        return self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    @torch.inference_mode()
    def score_batch(
        self,
        prompt_texts: list[str],
        candidates_by_example: list[list[dict[str, str]]],
        batch_size: int = 24,
    ) -> list[list[ScoredCandidate]]:
        prepared: list[tuple[int, str, str, str, list[int], list[int]]] = []
        for example_idx, (prompt_text, candidates) in enumerate(
            zip(prompt_texts, candidates_by_example, strict=True)
        ):
            prompt_ids = self.encode_prompt(prompt_text)
            for candidate in candidates:
                candidate_ids = self.encode_candidate(candidate["score_text"])
                prepared.append(
                    (
                        example_idx,
                        candidate["label"],
                        candidate["text"],
                        candidate["score_text"],
                        prompt_ids,
                        candidate_ids,
                    )
                )

        grouped: list[list[ScoredCandidate]] = [[] for _ in prompt_texts]
        device = self.model.device
        current_batch_size = max(1, batch_size)
        start = 0
        while start < len(prepared):
            batch = prepared[start : start + current_batch_size]
            sequences = [prompt_ids + candidate_ids for *_meta, prompt_ids, candidate_ids in batch]
            max_len = max(len(seq) for seq in sequences)
            pad_id = self.tokenizer.pad_token_id
            input_ids = []
            attention_mask = []
            for seq in sequences:
                padding = [pad_id] * (max_len - len(seq))
                input_ids.append(seq + padding)
                attention_mask.append([1] * len(seq) + [0] * len(padding))

            try:
                input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
                mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
                outputs = self.model(input_ids=input_tensor, attention_mask=mask_tensor)
                log_probs = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                continue

            for row_idx, entry in enumerate(batch):
                example_idx, label, text, score_text, prompt_ids, candidate_ids = entry
                prompt_len = len(prompt_ids)
                candidate_len = len(candidate_ids)
                token_logprobs = []
                for offset in range(candidate_len):
                    logit_pos = prompt_len - 1 + offset
                    target_id = input_tensor[row_idx, prompt_len + offset]
                    token_logprobs.append(log_probs[row_idx, logit_pos, target_id].item())
                logprob_sum = sum(token_logprobs)
                grouped[example_idx].append(
                    ScoredCandidate(
                        label=label,
                        text=text,
                        score_text=score_text,
                        logprob_sum=logprob_sum,
                        token_count=candidate_len,
                        first_token_logprob=token_logprobs[0],
                        length_norm_logprob=logprob_sum / max(candidate_len, 1),
                    )
                )
            start += len(batch)

        return grouped

    @torch.inference_mode()
    def extract_prompt_hidden_states(
        self,
        prompt_texts: list[str],
        layers: list[int],
        batch_size: int = 8,
    ) -> dict[int, torch.Tensor]:
        device = self.model.device
        collected = {layer: [] for layer in layers}
        for start in range(0, len(prompt_texts), batch_size):
            batch_prompts = prompt_texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = self.model(**encoded, output_hidden_states=True)
            final_positions = encoded["attention_mask"].sum(dim=1) - 1
            for layer in layers:
                hidden = outputs.hidden_states[layer]
                batch_vectors = hidden[torch.arange(hidden.size(0), device=device), final_positions]
                collected[layer].append(batch_vectors.detach().cpu())
        return {layer: torch.cat(chunks, dim=0) for layer, chunks in collected.items()}

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 4,
        batch_size: int = 8,
    ) -> list[str]:
        generations: list[str] = []
        device = self.model.device
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        current_batch_size = max(1, batch_size)
        start = 0
        try:
            while start < len(prompt_texts):
                batch_prompts = prompt_texts[start : start + current_batch_size]
                encoded = self.tokenizer(
                    batch_prompts,
                    padding=True,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                try:
                    outputs = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=50,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if current_batch_size == 1:
                        raise
                    current_batch_size = max(1, current_batch_size // 2)
                    continue
                prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                for sequence, prompt_len in zip(outputs, prompt_lengths, strict=True):
                    generated_ids = sequence[prompt_len:]
                    text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    generations.append(text.strip())
                start += len(batch_prompts)
        finally:
            self.tokenizer.padding_side = original_padding_side
        return generations
