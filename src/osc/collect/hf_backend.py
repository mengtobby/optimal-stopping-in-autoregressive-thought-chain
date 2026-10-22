"""Optional Transformers backend; model dependencies are imported only on construction."""

from collections.abc import Sequence

import numpy as np


class BackendError(ValueError):
    """Invalid model request or unsupported tokenizer/model."""


class HFBackend:
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        feature_layers: Sequence[int] = (-1,),
        temperature: float = 0.6,
        top_p: float = 0.95,
        seed: int = 0,
        score_chunk_size: int = 8,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        if temperature < 0 or not 0 < top_p <= 1 or not feature_layers:
            raise BackendError("invalid sampling parameters or empty feature_layers")
        if dtype not in ("float32", "float16", "bfloat16"):
            raise BackendError("dtype must be float32, float16, or bfloat16")
        if score_chunk_size < 1:
            raise BackendError("score_chunk_size must be positive")
        self.score_chunk_size = score_chunk_size
        set_seed(seed)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise BackendError("a fast tokenizer with character offsets is required")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=getattr(torch, dtype),
            device_map=device if device == "auto" else None,
            attn_implementation="eager",
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.feature_layers = tuple(feature_layers)
        self.temperature, self.top_p = temperature, top_p
        self._digit_token_ids = self._find_digit_token_ids()

    def _find_digit_token_ids(self) -> list[int]:
        return [
            i
            for i in range(len(self.tokenizer))
            if self.tokenizer.decode([i], skip_special_tokens=False)[:1].isdigit()
        ]

    def score_continuations(
        self,
        prompt: str,
        thinking_prefix: str,
        probe: str,
        candidates: Sequence[str],
        require_non_digit_next: bool = False,
    ) -> np.ndarray:
        """Score whole-string tokenizations; context must end at a token boundary."""
        context = self._prompt(prompt) + thinking_prefix + probe
        sequences, starts = [], []
        for candidate in candidates:
            encoded = self.tokenizer(
                context + candidate, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = encoded["offset_mapping"]
            if any(a < len(context) < b for a, b in offsets):
                raise BackendError("candidate straddles the context token boundary")
            indices = [i for i, (a, b) in enumerate(offsets) if a >= len(context) and b > a]
            start = indices[0] if indices else len(encoded["input_ids"])
            if start == 0:
                raise BackendError("context must contain at least one token")
            sequences.append(encoded["input_ids"])
            starts.append(start)
        if require_non_digit_next and not hasattr(self, "_digit_token_ids"):
            self._digit_token_ids = self._find_digit_token_ids()
        torch = self.torch
        scores = []
        chunk_size = getattr(self, "score_chunk_size", 8)
        for offset in range(0, len(sequences), chunk_size):
            chunk = sequences[offset : offset + chunk_size]
            width = max(map(len, chunk))
            ids = torch.zeros((len(chunk), width), dtype=torch.long, device=self.model.device)
            mask = torch.zeros_like(ids)
            for row, sequence in enumerate(chunk):
                ids[row, : len(sequence)] = torch.tensor(sequence, device=self.model.device)
                mask[row, : len(sequence)] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                for row, sequence in enumerate(chunk):
                    start, end = starts[offset + row], len(sequence)
                    positions = torch.arange(start, end, device=ids.device)
                    log_probs = logits[row, start - 1 : end - 1].float().log_softmax(-1)
                    score = log_probs.gather(-1, ids[row, positions, None]).sum()
                    if require_non_digit_next:
                        next_log_probs = logits[row, end - 1].float().log_softmax(-1)
                        allowed = torch.ones(
                            next_log_probs.shape[-1], dtype=torch.bool, device=ids.device
                        )
                        allowed[self._digit_token_ids] = False
                        score = score + next_log_probs[allowed].logsumexp(-1)
                    scores.append(float(score))
        return np.asarray(scores, dtype=float)

    def _prompt(self, prompt: str) -> str:
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        return text if text.endswith("<think>\n") else text + "<think>\n"

    def _sample(self, text: str, max_tokens: int, greedy: bool = False) -> str:
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        options = {"do_sample": not greedy and self.temperature > 0}
        if options["do_sample"]:
            options.update(temperature=self.temperature, top_p=self.top_p)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **options,
            )
        return self.tokenizer.decode(
            output[0, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )

    def generate_thinking(self, prompt: str, max_tokens: int) -> tuple[str, bool]:
        text = self._sample(self._prompt(prompt), max_tokens)
        return text.split("</think>", 1)[0], "</think>" in text

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def force_answers(self, prompt: str, thinking_prefix: str, n_samples: int) -> list[str]:
        if n_samples < 0:
            raise BackendError("n_samples must be nonnegative")
        if n_samples == 0:
            return []
        text = self._prompt(prompt) + thinking_prefix + "\n</think>\n\n**Final Answer**\n\\boxed{"
        encoded = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
            self.model.device
        )
        options = {"do_sample": self.temperature > 0}
        if options["do_sample"]:
            options.update(temperature=self.temperature, top_p=self.top_p)
        # Greedy generation cannot return multiple sequences; repeat its identical result.
        count = n_samples if options["do_sample"] else 1
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                max_new_tokens=256,
                num_return_sequences=count,
                pad_token_id=self.tokenizer.eos_token_id,
                **options,
            )
        answers = [
            "\\boxed{"
            + self.tokenizer.decode(row[encoded.input_ids.shape[1] :], skip_special_tokens=True)
            for row in output
        ]
        return answers if options["do_sample"] else answers * n_samples

    def continue_steps(
        self, prompt: str, thinking_prefix: str, k: int, max_tokens: int
    ) -> list[str]:
        return [
            self._sample(self._prompt(prompt) + thinking_prefix, max_tokens).split("</think>", 1)[0]
            for _ in range(k)
        ]

    def complete(self, prompt: str, thinking_prefix: str, probe: str, max_tokens: int) -> str:
        return self._sample(self._prompt(prompt) + thinking_prefix + probe, max_tokens, greedy=True)

    def _inputs(self, prompt: str, thinking: str, positions: Sequence[int]):
        if any(p < 0 or p > len(thinking) for p in positions):
            raise BackendError("character positions must lie within thinking")
        prefix = self._prompt(prompt)
        encoded = self.tokenizer(
            prefix + thinking,
            return_offsets_mapping=True,
            add_special_tokens=False,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        indices = []
        for position in positions:
            valid = [i for i, (a, b) in enumerate(offsets) if b <= len(prefix) + position and b > a]
            if not valid:
                raise BackendError("no token ends before requested character position")
            indices.append(valid[-1])
        return encoded.to(self.model.device), indices, offsets, len(prefix)

    def hidden_features(
        self, prompt: str, thinking: str, char_positions: Sequence[int]
    ) -> np.ndarray:
        inputs, indices, _, _ = self._inputs(prompt, thinking, char_positions)
        with self.torch.inference_mode():
            output = self.model(**inputs, output_hidden_states=True, use_cache=False)
        features = self.torch.cat(
            [output.hidden_states[layer][0, indices] for layer in self.feature_layers], dim=-1
        )
        return features.float().cpu().numpy()

    def attention_to_span(
        self, prompt: str, thinking: str, span: tuple[int, int], query_char_positions: Sequence[int]
    ) -> np.ndarray:
        # Intercept eager softmax inputs: only selected rows are materialized for the
        # probe, while SDPA computes the full forward output without an attention matrix.
        try:
            from transformers import AttentionInterface
        except ImportError as exc:
            raise BackendError("attention_to_span requires transformers>=4.51") from exc

        if not 0 <= span[0] <= span[1] <= len(thinking):
            raise BackendError("span must lie within thinking")
        inputs, queries, offsets, shift = self._inputs(prompt, thinking, query_char_positions)
        keys = [
            i
            for i, (a, b) in enumerate(offsets)
            if b > shift + span[0] and a < shift + span[1] and b > a
        ]
        torch = self.torch
        masses = []

        def attention(
            module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs
        ):
            repeats = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)
            scale = scaling if scaling is not None else query.shape[-1] ** -0.5
            selected = query[:, :, queries, :]
            logits = (selected @ key.transpose(-1, -2)) * scale
            qids = torch.tensor(queries, device=query.device)
            kids = torch.arange(key.shape[-2], device=query.device)
            logits = logits.masked_fill(kids[None, :] > qids[:, None], float("-inf"))
            if attention_mask is not None:
                logits = logits + attention_mask[:, :, queries, : key.shape[-2]]
            weights = torch.softmax(logits.float(), dim=-1)
            masses.append(weights[..., keys].sum(-1)[0].transpose(0, 1).cpu().numpy())
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                scale=scale,
            )
            return output.transpose(1, 2).contiguous(), None

        name = "osc_selected_queries"
        AttentionInterface.register(name, attention)
        configs = []
        handles = []

        # Hooks scope the custom attention implementation to self-attention modules.
        def before(module, args):
            configs.append((module.config, module.config._attn_implementation))
            module.config._attn_implementation = name

        for module in self.model.modules():
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "config")
            ):
                handles.append(module.register_forward_pre_hook(before))
        try:
            with torch.inference_mode():
                self.model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            for config, implementation in reversed(configs):
                config._attn_implementation = implementation
        if not masses:
            raise BackendError("model does not expose compatible self-attention modules")
        return np.stack(masses, axis=1)
