"""Message conversion helpers for DeepSeek chat models."""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    InvalidToolCall,
    SystemMessage,
    SystemMessageChunk,
    ToolCall,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.output_parsers.openai_tools import (
    make_invalid_tool_call,
    parse_tool_call,
)

from langchain_deepseek.chat_models._utils import _create_usage_metadata


def _format_tool_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _convert_message_to_dict(message: BaseMessage) -> dict[str, Any]:  # noqa: C901
    """Convert a LangChain message to a DeepSeek chat completions message dict."""
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, AIMessage):
        content = message.content
        if isinstance(content, list):
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            content = text
        message_dict = {"role": "assistant", "content": content or None}
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_deepseek_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_deepseek_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
        if "reasoning_content" in message.additional_kwargs:
            message_dict["reasoning_content"] = message.additional_kwargs[
                "reasoning_content"
            ]
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": _format_tool_content(message.content),
            "tool_call_id": message.tool_call_id,
        }
    else:
        msg = f"Got unknown message type {message!r}"
        raise TypeError(msg)
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:  # noqa: C901
    """Convert a DeepSeek response message dict to a LangChain message."""
    role = _dict.get("role")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""))
    if role == "system":
        return SystemMessage(content=_dict.get("content", ""))
    if role == "tool":
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id", ""),
        )
    if role == "assistant":
        additional_kwargs: dict[str, Any] = {}
        if "reasoning_content" in _dict and _dict.get("reasoning_content") is not None:
            additional_kwargs["reasoning_content"] = _dict["reasoning_content"]
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls

        tool_calls = []
        invalid_tool_calls = []
        for raw_tool_call in _dict.get("tool_calls") or []:
            try:
                tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
            except Exception as e:  # noqa: BLE001, PERF203
                invalid_tool_calls.append(make_invalid_tool_call(raw_tool_call, str(e)))
        return AIMessage(
            content=_dict.get("content") or "",
            id=_dict.get("id"),
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={"model_provider": "deepseek"},
        )
    if role is None:
        msg = f"DeepSeek response message is missing role. Keys: {list(_dict.keys())}"
        raise ValueError(msg)
    warnings.warn(
        f"Unrecognized message role {role!r} from DeepSeek; using ChatMessage.",
        stacklevel=2,
    )
    return ChatMessage(content=_dict.get("content", ""), role=role)


def _convert_chunk_to_message_chunk(
    chunk: Mapping[str, Any],
    default_class: type[BaseMessageChunk],
) -> BaseMessageChunk:
    """Convert a DeepSeek streaming chunk into a LangChain message chunk."""
    choice = chunk["choices"][0]
    delta = choice.get("delta", {})
    role = cast("str | None", delta.get("role"))
    content = cast("str", delta.get("content") or "")

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    if role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    if role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content,
            tool_call_id=delta.get("tool_call_id", ""),
        )
    if role and role != "assistant":
        return ChatMessageChunk(content=content, role=role)

    additional_kwargs: dict[str, Any] = {}
    if "reasoning_content" in delta and delta.get("reasoning_content") is not None:
        additional_kwargs["reasoning_content"] = delta["reasoning_content"]

    tool_call_chunks = []
    for raw_tool_call in delta.get("tool_calls") or []:
        try:
            tool_call_chunks.append(
                tool_call_chunk(
                    name=raw_tool_call["function"].get("name"),
                    args=raw_tool_call["function"].get("arguments"),
                    id=raw_tool_call.get("id"),
                    index=raw_tool_call["index"],
                )
            )
        except (KeyError, TypeError, AttributeError):  # noqa: PERF203
            warnings.warn(
                f"Skipping malformed DeepSeek tool call chunk: {raw_tool_call!r}",
                stacklevel=2,
            )

    usage_metadata = None
    if usage := chunk.get("usage"):
        usage_metadata = _create_usage_metadata(usage)
    return AIMessageChunk(
        content=content,
        additional_kwargs=additional_kwargs,
        tool_call_chunks=tool_call_chunks,  # type: ignore[arg-type]
        usage_metadata=usage_metadata,  # type: ignore[arg-type]
        response_metadata={"model_provider": "deepseek"},
    )


def _lc_tool_call_to_deepseek_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call["id"],
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
        },
    }


def _lc_invalid_tool_call_to_deepseek_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict[str, Any]:
    return {
        "id": invalid_tool_call["id"],
        "type": "function",
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }
