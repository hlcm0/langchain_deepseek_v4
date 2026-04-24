"""Utility helpers for DeepSeek chat models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages.ai import (
    InputTokenDetails,
    OutputTokenDetails,
    UsageMetadata,
)
from langchain_core.utils.pydantic import is_basemodel_subclass

_INTERNAL_KWARGS = frozenset({"ls_structured_output_format"})


def _dump_response(response: Any) -> dict[str, Any]:
    """Convert an SDK response object to a dictionary."""
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True, by_alias=True)
    if hasattr(response, "dict"):
        return response.dict()
    msg = f"Unsupported DeepSeek response type: {type(response)!r}"
    raise TypeError(msg)


def _strip_internal_kwargs(params: dict[str, Any]) -> None:
    """Remove LangChain-internal kwargs before passing params to the SDK."""
    for key in _INTERNAL_KWARGS:
        params.pop(key, None)


def _normalize_payload_for_openai_sdk(params: dict[str, Any]) -> None:
    """Move DeepSeek-only top-level params into `extra_body` for OpenAI SDK."""
    _strip_internal_kwargs(params)
    extra_body = dict(params.get("extra_body") or {})
    if thinking := params.pop("thinking", None):
        extra_body["thinking"] = thinking
    if extra_body:
        params["extra_body"] = extra_body


def _create_usage_metadata(token_usage: Mapping[str, Any]) -> UsageMetadata:
    """Create LangChain usage metadata from a DeepSeek usage payload."""
    input_tokens = int(token_usage.get("prompt_tokens") or 0)
    output_tokens = int(token_usage.get("completion_tokens") or 0)
    total_tokens = int(token_usage.get("total_tokens") or input_tokens + output_tokens)

    input_details: dict[str, int] = {}
    if cache_hit := token_usage.get("prompt_cache_hit_tokens"):
        input_details["cache_read"] = int(cache_hit)

    output_details: dict[str, int] = {}
    completion_details = token_usage.get("completion_tokens_details") or {}
    if reasoning := completion_details.get("reasoning_tokens"):
        output_details["reasoning"] = int(reasoning)

    usage_metadata: UsageMetadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    if input_details:
        usage_metadata["input_token_details"] = InputTokenDetails(**input_details)  # type: ignore[typeddict-item]
    if output_details:
        usage_metadata["output_token_details"] = OutputTokenDetails(**output_details)  # type: ignore[typeddict-item]
    return usage_metadata


def _is_pydantic_class(obj: Any) -> bool:
    """Return whether an object is a Pydantic model class."""
    return isinstance(obj, type) and is_basemodel_subclass(obj)
