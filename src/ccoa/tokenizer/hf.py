"""Load and apply a HuggingFace tokenizer to plain text.

Only the *fast* (Rust-backed) `PreTrainedTokenizerFast` variants are
supported. Fast tokenizers are thread-safe and release the GIL during
encoding, so the `tokenize` command can share a single tokenizer
instance across worker threads without a lock.

Gated repos (e.g. `meta-llama/Llama-2-7b`) require a HuggingFace access
token; the underlying `huggingface_hub` library picks up `HF_TOKEN` from
the environment or `~/.cache/huggingface/token` automatically. Accept
the model's license on the HuggingFace web UI first, then either
`export HF_TOKEN=...` or run `huggingface-cli login`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

DEFAULT_TOKENIZER_REPO = "meta-llama/Llama-2-7b"


def load_tokenizer(repo: str = DEFAULT_TOKENIZER_REPO) -> PreTrainedTokenizerBase:
    """Download (cached) and load a fast HuggingFace tokenizer from `repo`.

    Raises `RuntimeError` if the resolved tokenizer is the slow Python
    variant — those are not thread-safe and the `tokenize` command's
    thread mode would race.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            f"Tokenizer for {repo!r} resolved to the slow Python variant "
            f"({type(tokenizer).__name__}); fast tokenizers are required for "
            f"thread-mode safety. Install the matching fast-tokenizer extra "
            f"for this model, or run with --workers-mode process."
        )
    return tokenizer


def tokenize_batch(tokenizer: PreTrainedTokenizerBase, texts: list[str]) -> list[list[int]]:
    """Tokenize `texts` and return one list of token ids per input.

    No special tokens, no padding, no truncation — the caller wants raw
    token sequences for counting and length analysis.
    """
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return [list(ids) for ids in encoded["input_ids"]]
