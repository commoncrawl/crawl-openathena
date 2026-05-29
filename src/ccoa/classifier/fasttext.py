"""Load and apply a HuggingFace-hosted fasttext classifier to a text snippet."""

from __future__ import annotations

from collections.abc import Sequence

import fasttext as _fasttext
from huggingface_hub import hf_hub_download

DEFAULT_MODEL_REPO = "ibm-granite/GneissWeb.Sci_classifier"
DEFAULT_MODEL_FILE = "fasttext_science.bin"

FASTTEXT_MAX_INPUT_CHARS = 100_000


def load_classifier(
    model_repo: str = DEFAULT_MODEL_REPO,
    model_file: str = DEFAULT_MODEL_FILE,
):
    """Download (cached) the model file from HuggingFace Hub and load it."""
    path = hf_hub_download(repo_id=model_repo, filename=model_file)
    return _fasttext.load_model(path)


def clean_for_fasttext(text: str, max_len: int = FASTTEXT_MAX_INPUT_CHARS) -> str:
    """Normalise text into a single safe line that fasttext's C buffer accepts.

    - Replaces `\\n` / `\\r` with spaces (fasttext's `predict` rejects newlines).
    - Strips NUL bytes (`\\x00`) — some fasttext builds segfault on them.
    - Clamps length to `max_len` characters. Web pages often run into millions
      of characters of boilerplate; the classifier signal saturates long before
      that and overly long inputs have been blamed for heap-corruption aborts.
    """
    cleaned = text.replace("\n", " ").replace("\r", " ").replace("\x00", "")
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def predict_targets(model, text: str, target_labels: Sequence[str]) -> list[float]:
    """Return probabilities for `target_labels` in the order given.

    `model.predict(..., k=-1)` returns labels in descending probability order
    (data-dependent), so we resolve scores by label name. Missing labels
    yield 0.0.
    """
    cleaned = clean_for_fasttext(text)
    labels, probs = model.predict(cleaned, k=-1)
    label_to_prob = dict(zip(labels, probs, strict=False))
    return [float(label_to_prob.get(lbl, 0.0)) for lbl in target_labels]


def get_model_labels(model) -> tuple[str, ...]:
    """Return all labels the model can emit, in model-internal order."""
    return tuple(model.get_labels())
