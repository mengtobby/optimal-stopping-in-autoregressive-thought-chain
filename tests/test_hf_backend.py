import pytest

pytestmark = pytest.mark.model


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
    monkeypatch, device, capabilities, expected, checked
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
    torch = SimpleNamespace(cuda=cuda, float16="float16", bfloat16="bfloat16")
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
    if device != "auto":
        model.to.assert_called_once_with(device)
