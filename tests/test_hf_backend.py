import pytest

pytestmark = pytest.mark.model


@pytest.mark.parametrize("masked", [False, True])
def test_probe_logits_float32_before_matmul(masked):
    from types import SimpleNamespace

    import numpy as np

    from osc.collect.hf_backend import _probe_logits

    class Tensor:
        def __init__(self, data):
            self.data = np.asarray(data)
            self.shape = self.data.shape
            self.device = "cpu"

        def __getitem__(self, item):
            return Tensor(self.data[item])

        def float(self):
            return Tensor(self.data.astype(np.float32))

        def transpose(self, a, b):
            return Tensor(self.data.swapaxes(a, b))

        def __matmul__(self, other):
            assert self.data.dtype == other.data.dtype == np.float32
            return Tensor(self.data @ other.data)

        def __mul__(self, scale):
            return Tensor(self.data * scale)

        def __gt__(self, other):
            return self.data > other.data

        def masked_fill(self, mask, value):
            assert self.data.dtype == np.float32
            return Tensor(np.where(mask, value, self.data))

        def __add__(self, other):
            assert self.data.dtype == other.data.dtype == np.float32
            return Tensor(self.data + other.data)

    torch = SimpleNamespace(
        tensor=lambda data, **kwargs: Tensor(data),
        arange=lambda n, **kwargs: Tensor(np.arange(n)),
    )
    # The unscaled dot product exceeds the largest finite float16 value.
    query = Tensor(np.full((1, 1, 2, 2), 256, dtype=np.float16))
    key = Tensor(np.full((1, 1, 3, 2), 256, dtype=np.float16))
    mask = Tensor(np.full((1, 1, 2, 3), -2, dtype=np.float16)) if masked else None
    logits = _probe_logits(torch, query, key, [0, 1], mask, 0.5).data
    assert logits.dtype == np.float32
    np.testing.assert_array_equal(logits[0, 0, 1, :2], [65536 - 2 * masked] * 2)
    assert np.isneginf(logits[0, 0, 0, 1:]).all()


@pytest.fixture
def tiny_backend():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizers = pytest.importorskip("tokenizers")
    from osc.collect.hf_backend import HFBackend

    torch.manual_seed(0)
    vocabulary = {"[UNK]": 0, "q": 1, "a": 2, "b": 3, "1": 4, "12": 5, " 3": 6}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend = HFBackend.__new__(HFBackend)
    backend.torch = torch
    backend.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    backend.tokenizer.chat_template = (
        "{% for message in messages %}{{message['content']}} {% endfor %}"
    )
    backend.model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(vocabulary),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            attn_implementation="eager",
        )
    ).eval()
    backend.feature_layers = (-1,)
    backend.temperature, backend.top_p = 0.6, 0.95
    return backend


def test_tiny_model_features_and_attention(tiny_backend):
    backend = tiny_backend
    thinking = " a b"
    assert backend.hidden_features("q", thinking, [0, 2, 4]).shape == (3, 16)
    masses = backend.attention_to_span("q", thinking, (1, 2), [0, 2, 4])
    assert masses.shape == (3, 2, 2)
    assert (masses[0] == 0).all()
    assert ((masses >= 0) & (masses <= 1)).all()


@pytest.mark.parametrize("non_digit", [False, True])
def test_score_continuations_matches_manual(tiny_backend, non_digit, monkeypatch):
    import numpy as np

    backend = tiny_backend
    torch = backend.torch
    candidates = [" 1", " 12", " a b", ""] * 3
    thinking = "a " * 32 + "a"
    context = backend._prompt("q") + thinking + " "
    expected = []
    for candidate in candidates:
        encoded = backend.tokenizer(
            context + candidate,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.inference_mode():
            log_probs = backend.model(**encoded).logits[0].float().log_softmax(-1)
        score = sum(
            float(log_probs[i - 1, encoded.input_ids[0, i]])
            for i, (a, b) in enumerate(offsets)
            if a >= len(context) and b > a
        )
        if non_digit:
            digit_ids = [4, 5]
            p_digit = log_probs[-1, digit_ids].exp().sum()
            score += float(torch.log1p(-p_digit))
        expected.append(score)
    calls = []
    forward = backend.model.forward

    def record_forward(**kwargs):
        calls.append(
            (
                tuple(kwargs["input_ids"].shape),
                kwargs.get("logits_to_keep"),
                kwargs.get("past_key_values").get_seq_length()
                if kwargs.get("past_key_values") is not None
                else 0,
            )
        )
        assert kwargs["use_cache"] is True
        assert backend.model.config._attn_implementation == "sdpa"
        return forward(**kwargs)

    monkeypatch.setattr(backend.model, "forward", record_forward)
    actual = backend.score_continuations("q", thinking, " ", candidates, non_digit)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    context_length = len(backend.tokenizer.encode(context, add_special_tokens=False))
    assert calls == [
        ((1, context_length), 1, 0),
        ((8, 2), None, context_length),
        ((4, 2), None, context_length),
    ]
    assert backend.model.config._attn_implementation == "eager"
    assert backend._find_digit_token_ids() == [4, 5]
    assert backend.score_continuations("q", "", "", []).shape == (0,)


def test_score_rejects_changed_context_ids(tiny_backend, monkeypatch):
    from osc.collect.hf_backend import BackendError

    monkeypatch.setattr(tiny_backend.tokenizer, "encode", lambda *args, **kwargs: [2])
    with pytest.raises(BackendError, match="changes the context"):
        tiny_backend.score_continuations("q", "", "", [" 1"])


def test_score_empty_suffixes_and_attention_restoration(tiny_backend, monkeypatch):
    import numpy as np

    backend = tiny_backend
    np.testing.assert_array_equal(backend.score_continuations("q", "", "", ["", ""]), [0, 0])

    def fail(**kwargs):
        raise RuntimeError("prefill failed")

    monkeypatch.setattr(backend.model, "forward", fail)
    with pytest.raises(RuntimeError, match="prefill failed"):
        backend.score_continuations("q", "", "", [" 1"])
    assert backend.model.config._attn_implementation == "eager"


def test_score_boundary_and_digit_penalty(tiny_backend):
    import numpy as np

    from osc.collect.hf_backend import BackendError

    backend = tiny_backend
    with pytest.raises(BackendError, match="boundary"):
        backend.score_continuations("q", "a", "", ["b"])
    plain = backend.score_continuations("q", "", "", [" 1"])
    penalized = backend.score_continuations("q", "", "", [" 1"], True)
    encoded = backend.tokenizer(
        backend._prompt("q") + " 1", return_tensors="pt", add_special_tokens=False
    )
    with backend.torch.inference_mode():
        probs = backend.model(**encoded).logits[0, -1].float().softmax(-1)
    np.testing.assert_allclose(
        penalized - plain, [float(backend.torch.log1p(-probs[[4, 5]].sum()))], atol=1e-6
    )


def test_force_answers_one_generate_call(tiny_backend, monkeypatch):
    backend = tiny_backend
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        ids = kwargs["input_ids"].repeat(kwargs["num_return_sequences"], 1)
        suffix = backend.torch.full((len(ids), 1), 4, dtype=ids.dtype)
        return backend.torch.cat([ids, suffix], dim=1)

    monkeypatch.setattr(backend.model, "generate", generate)
    assert backend.force_answers("q", "a", 3) == ["\\boxed{1"] * 3
    assert len(calls) == 1
    assert calls[0]["num_return_sequences"] == 3
    assert calls[0]["do_sample"] is True


def test_force_answers_real_batched_generation(tiny_backend, monkeypatch):
    backend = tiny_backend
    generate = backend.model.generate
    calls = []

    def short_generate(**kwargs):
        calls.append(kwargs["num_return_sequences"])
        kwargs["max_new_tokens"] = 2
        return generate(**kwargs)

    monkeypatch.setattr(backend.model, "generate", short_generate)
    answers = backend.force_answers("q", "a", 3)
    assert len(answers) == 3
    assert all(isinstance(answer, str) and answer.startswith("\\boxed{") for answer in answers)
    assert calls == [3]
    backend.temperature = 0
    assert len(set(backend.force_answers("q", "a", 3))) == 1
    assert calls == [3, 1]
    assert backend.force_answers("q", "a", 0) == []


def test_attention_version_guard(monkeypatch):
    import sys
    import types

    from osc.collect.hf_backend import BackendError, HFBackend

    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    with pytest.raises(BackendError, match=r"transformers>=4\.51"):
        HFBackend.__new__(HFBackend).attention_to_span("q", "a", (0, 1), [1])


def test_non_eager_attention_rejected(tiny_backend):
    from osc.collect.hf_backend import BackendError

    tiny_backend.attn_implementation = "sdpa"
    with pytest.raises(BackendError, match="requires loading.*eager"):
        tiny_backend.attention_to_span("q", "a", (0, 1), [1])


def test_constructor_auto_cpu_and_lazy_digits(tiny_backend, monkeypatch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from osc.collect.hf_backend import HFBackend

    calls = []
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tiny_backend.tokenizer)

    def load(*args, **kwargs):
        calls.append(kwargs)
        return tiny_backend.model

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    decode = tiny_backend.tokenizer.decode
    decoded = []

    def record(*args, **kwargs):
        decoded.append(args)
        return decode(*args, **kwargs)

    monkeypatch.setattr(tiny_backend.tokenizer, "decode", record)
    backend = HFBackend("tiny", device="cpu", dtype="auto", attn_implementation="sdpa")
    assert calls[0]["dtype"] == tiny_backend.torch.float32
    assert calls[0]["attn_implementation"] == "sdpa"
    assert decoded == []
    backend.score_continuations("q", "", "", [" 1"])
    assert len(decoded) == len(backend.tokenizer)
    backend.score_continuations("q", "", "", [" 1"])
    assert len(decoded) == len(backend.tokenizer)


@pytest.mark.parametrize(
    "device,capabilities,expected,checked",
    [
        ("auto", [], "float32", []),
        ("auto", [(7, 5)], "float16", [0]),
        ("auto", [(6, 0)], "float16", [0]),
        ("auto", [(8, 0)], "bfloat16", [0]),
        ("auto", [(8, 0), (7, 5)], "float16", [0, 1]),
        ("cuda:1", [(8, 0), (7, 5)], "float16", [1]),
        ("cuda:0", [(8, 0), (7, 5)], "bfloat16", [0]),
        ("cuda", [(7, 5)], "float16", [None]),
    ],
)
def test_auto_dtype_uses_native_cuda_capability(
    monkeypatch, device, capabilities, expected, checked, caplog
):
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock

    from osc.collect.hf_backend import HFBackend

    def capability(index):
        return capabilities[0 if index is None else index]

    cuda = SimpleNamespace(
        is_available=lambda: True,
        is_bf16_supported=lambda: True,  # Emulation must not select bf16 on sm75/sm60.
        device_count=lambda: len(capabilities),
        get_device_capability=Mock(side_effect=capability),
    )
    torch = SimpleNamespace(cuda=cuda, float16="float16", bfloat16="bfloat16", float32="float32")
    model = Mock()
    loader = Mock(return_value=model)
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *a, **kw: SimpleNamespace(is_fast=True)
        ),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
        set_seed=lambda seed: None,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    HFBackend("stub", device=device, dtype="auto")
    assert loader.call_args.kwargs["dtype"] == expected
    assert loader.call_args.kwargs["device_map"] == ("auto" if device == "auto" else None)
    assert [call.args[0] for call in cuda.get_device_capability.call_args_list] == checked
    if not capabilities:
        assert "no devices were enumerated" in caplog.text
    if device != "auto":
        model.to.assert_called_once_with(device)


@pytest.mark.parametrize("text,count", [("1}later", 2), (r"\frac{1}{2}}x", 12), ("123", 3)])
def test_answer_confidence_closing_brace(text, count):
    from types import SimpleNamespace

    import numpy as np

    from osc.collect.hf_backend import HFBackend

    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "".join(ids))
    scores = -np.arange(1, len(text) + 1, dtype=float)
    assert backend._answer_confidence(list(text), scores) == pytest.approx(
        np.exp(np.mean(scores[:count]))
    )


def test_scored_generation_without_model():
    from contextlib import nullcontext
    from types import SimpleNamespace

    import numpy as np

    from osc.collect.hf_backend import BackendError, HFBackend

    class Encoded(dict):
        def __init__(self):
            self.input_ids = np.array([[9, 9]])
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            return self

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, *args, **kwargs):
            return Encoded()

        def decode(self, ids, **kwargs):
            return "".join({1: "1", 2: "}", 3: "tail"}[int(i)] for i in ids)

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        sequences = np.tile([9, 9, 1, 2, 3], (kwargs["num_return_sequences"], 1))
        if kwargs.get("output_logits"):
            assert kwargs["return_dict_in_generate"] and "output_scores" not in kwargs
            return SimpleNamespace(sequences=sequences, logits="logits")
        return sequences

    def transition(sequences, scores, normalize_logits):
        assert scores == "logits" and normalize_logits
        return np.tile(np.log([0.25, 1, 0.01]), (len(sequences), 1))

    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer = Tokenizer()
    backend.torch = SimpleNamespace(inference_mode=nullcontext)
    backend.model = SimpleNamespace(
        device="cpu", generate=generate, compute_transition_scores=transition
    )
    backend._prompt = lambda prompt: prompt
    backend.temperature, backend.top_p = 0.6, 0.95
    assert backend.force_answers("q", "", 2) == [r"\boxed{1}tail"] * 2
    assert "output_logits" not in calls[-1]
    assert backend.force_answers_scored("q", "", 2) == [(r"\boxed{1}tail", 0.5)] * 2
    assert calls[-1]["num_return_sequences"] == 2
    backend.temperature = 0
    assert backend.force_answers_scored("q", "", 3) == [(r"\boxed{1}tail", 0.5)] * 3
    assert calls[-1]["num_return_sequences"] == 1
    assert backend.force_answers_scored("q", "", 0) == []
    with pytest.raises(BackendError, match="nonnegative"):
        backend.force_answers_scored("q", "", -1)


def test_answer_confidence_invalid_and_zero_probability():
    from types import SimpleNamespace

    from osc.collect.hf_backend import BackendError, HFBackend

    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer = SimpleNamespace(decode=lambda ids, **kw: "1}")
    assert backend._answer_confidence([1], [float("-inf")]) == 0
    for scores in ([], [0.1], [float("nan")]):
        with pytest.raises(BackendError):
            backend._answer_confidence([1], scores)


def test_unclosed_answer_confidence_stops_at_eos():
    from types import SimpleNamespace

    import numpy as np

    from osc.collect.hf_backend import HFBackend

    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer = SimpleNamespace(
        eos_token_id=0, decode=lambda ids, **kw: "".join("x" if i else "" for i in ids)
    )
    scores = np.array([-1.0, -2.0, -3.0, float("-inf")])
    # The EOS token itself is scored; the padding after it is not.
    assert backend._answer_confidence([5, 6, 0, 0], scores) == pytest.approx(np.exp(-2.0))
    assert backend._answer_confidence([5, 6, 7], scores[:3]) == pytest.approx(np.exp(-2.0))
