import pytest

pytestmark = pytest.mark.model


def test_tiny_model_features_and_attention():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    tokenizers = pytest.importorskip("tokenizers")
    from osc.collect.hf_backend import HFBackend

    vocabulary = {"[UNK]": 0, "q": 1, "a": 2, "b": 3}
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
            vocab_size=4,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            attn_implementation="eager",
        )
    ).eval()
    backend.feature_layers = (-1,)
    thinking = " a b"
    assert backend.hidden_features("q", thinking, [0, 2, 4]).shape == (3, 16)
    masses = backend.attention_to_span("q", thinking, (1, 2), [0, 2, 4])
    assert masses.shape == (3, 2, 2)
    assert (masses[0] == 0).all()
    assert ((masses >= 0) & (masses <= 1)).all()
