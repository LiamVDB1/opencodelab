from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

_HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ModelSettings:
    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    speculative_mtp_tokens: int = 0
    language_model_only: bool = False


def normalize_huggingface_model(value: str) -> str:
    raw = value.strip()
    if any(char in raw for char in "\r\n\t\0"):
        raise ValueError("Hugging Face model id must be a single line")

    if raw.startswith(("https://huggingface.co/", "https://hf.co/")):
        parsed = urlparse(raw)
        if parsed.netloc not in {"huggingface.co", "hf.co"}:
            raise ValueError("Only huggingface.co or hf.co model URLs are supported")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 2:
            raise ValueError("Expected a Hugging Face model URL like https://huggingface.co/owner/model")
        raw = f"{parts[0]}/{parts[1]}"
    elif "://" in raw:
        raise ValueError("Only Hugging Face model URLs are supported")

    if not _HF_ID.fullmatch(raw):
        raise ValueError("Expected a Hugging Face model id like owner/model")
    return raw


def safe_slug(value: str, *, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = slug[:max_length].rstrip("-")
    if not slug:
        raise ValueError("Could not derive a safe profile name")
    return slug


def infer_model_settings(model_id: str) -> ModelSettings:
    lower = model_id.lower()
    if "qwen3.8" in lower:
        return ModelSettings(
            tool_call_parser="qwen3_coder",
            reasoning_parser="qwen3",
            speculative_mtp_tokens=3,
            language_model_only=True,
        )
    if "qwen3" in lower:
        return ModelSettings(tool_call_parser="qwen3_coder", reasoning_parser="qwen3")
    if "qwen2.5" in lower or "qwq" in lower:
        return ModelSettings(tool_call_parser="hermes")
    return ModelSettings()
