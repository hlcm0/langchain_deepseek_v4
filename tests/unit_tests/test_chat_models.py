"""Unit tests for ChatDeepSeek."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, Field, SecretStr

from langchain_deepseek import ChatDeepSeek
from langchain_deepseek.chat_models._convert import (
    _convert_chunk_to_message_chunk,
    _convert_message_to_dict,
)


class GetWeather(BaseModel):
    """Get weather for a location."""

    location: str = Field(description="City name")


class Answer(BaseModel):
    """Structured answer."""

    answer: str
    confidence: int


class _MockClient:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


class _MockStreamClient(_MockClient):
    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return iter(self.response)


def _make_model(response: Any | None = None, *, stream: bool = False) -> ChatDeepSeek:
    model = ChatDeepSeek(model="deepseek-v4-pro", api_key=SecretStr("test-key"))
    model.client = _MockStreamClient(response or []) if stream else _MockClient(response)
    return model


def test_tool_calling_parses_tool_calls() -> None:
    response = {
        "id": "chatcmpl_1",
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "I need the weather tool.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "GetWeather",
                                "arguments": '{"location": "Hangzhou"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }
    model = _make_model(response)

    msg = model.bind_tools([GetWeather]).invoke("weather?")

    assert msg.tool_calls == [
        {"name": "GetWeather", "args": {"location": "Hangzhou"}, "id": "call_1", "type": "tool_call"}
    ]
    assert msg.additional_kwargs["reasoning_content"] == "I need the weather tool."
    assert model.client.calls[0]["tools"][0]["function"]["name"] == "GetWeather"


def test_native_json_structured_output_parses_pydantic() -> None:
    response = {
        "id": "chatcmpl_2",
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": '{"answer": "yes", "confidence": 9}',
                },
            }
        ],
    }
    model = _make_model(response)

    parsed = model.with_structured_output(Answer, method="json_mode").invoke(
        "Return json with answer and confidence."
    )

    assert parsed == Answer(answer="yes", confidence=9)
    assert model.client.calls[0]["response_format"] == {"type": "json_object"}


def test_multi_turn_payload_preserves_reasoning_content_for_tool_calls() -> None:
    ai_message = AIMessage(
        content="",
        additional_kwargs={"reasoning_content": "Need to call weather."},
        tool_calls=[
            {
                "name": "GetWeather",
                "args": {"location": "Hangzhou"},
                "id": "call_1",
                "type": "tool_call",
            }
        ],
    )
    payload = [
        _convert_message_to_dict(HumanMessage(content="weather?")),
        _convert_message_to_dict(ai_message),
        _convert_message_to_dict(ToolMessage(content="24C", tool_call_id="call_1")),
        _convert_message_to_dict(HumanMessage(content="summarize")),
    ]

    assert payload[1]["reasoning_content"] == "Need to call weather."
    assert payload[1]["tool_calls"][0]["function"]["arguments"] == '{"location": "Hangzhou"}'
    assert payload[2] == {"role": "tool", "content": "24C", "tool_call_id": "call_1"}


def test_reasoning_content_is_extracted_from_response() -> None:
    response = {
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "9.11 has hundredths, 9.8 is 9.80.",
                    "content": "9.8 is greater.",
                },
            }
        ],
    }
    model = _make_model(response)

    msg = model.invoke("9.11 and 9.8, which is greater?")

    assert msg.content == "9.8 is greater."
    assert msg.additional_kwargs["reasoning_content"] == "9.11 has hundredths, 9.8 is 9.80."


def test_streaming_reasoning_and_content_chunks() -> None:
    model = _make_model(
        [
            {
                "id": "chunk_1",
                "model": "deepseek-v4-pro",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "reasoning_content": "Think"}}
                ],
            },
            {
                "id": "chunk_1",
                "model": "deepseek-v4-pro",
                "choices": [{"index": 0, "delta": {"reasoning_content": "ing"}}],
            },
            {
                "id": "chunk_1",
                "model": "deepseek-v4-pro",
                "choices": [{"index": 0, "delta": {"content": "Answer"}}],
            },
            {
                "id": "chunk_1",
                "model": "deepseek-v4-pro",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ],
        stream=True,
    )

    chunks = list(model.stream("stream please"))

    assert isinstance(chunks[0], AIMessageChunk)
    assert chunks[0].additional_kwargs["reasoning_content"] == "Think"
    assert chunks[1].additional_kwargs["reasoning_content"] == "ing"
    assert chunks[2].content == "Answer"
    assert model.client.calls[0]["stream"] is True


def test_direct_chunk_conversion_supports_reasoning_content() -> None:
    chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "reasoning_content": "because"},
            }
        ]
    }

    result = _convert_chunk_to_message_chunk(chunk, AIMessageChunk)

    assert result.additional_kwargs["reasoning_content"] == "because"


def test_thinking_is_sent_through_extra_body_for_openai_sdk() -> None:
    response = {
        "model": "deepseek-v4-pro",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
    }
    model = ChatDeepSeek(
        model="deepseek-v4-pro",
        api_key=SecretStr("test-key"),
        thinking={"type": "enabled"},
        reasoning_effort="high",
    )
    model.client = _MockClient(response)

    model.invoke("hello")

    assert model.client.calls[0]["extra_body"] == {"thinking": {"type": "enabled"}}
    assert model.client.calls[0]["reasoning_effort"] == "high"
