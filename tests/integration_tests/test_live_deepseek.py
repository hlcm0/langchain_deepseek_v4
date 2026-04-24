"""Live DeepSeek smoke tests.

These tests are skipped unless DEEPSEEK_API_KEY is set.
"""

from __future__ import annotations

import os

import pytest
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_deepseek import ChatDeepSeek

pytestmark = pytest.mark.integration


class JsonAnswer(BaseModel):
    """Tiny structured response."""

    answer: str = Field(description="Short answer")


def _live_model(**kwargs):
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is not set")
    return ChatDeepSeek(model="deepseek-v4-pro", max_tokens=256, **kwargs)


def test_live_tool_calling() -> None:
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"{location}: sunny"

    llm = _live_model(thinking={"type": "disabled"}).bind_tools([get_weather])
    msg = llm.invoke("Use the tool to get weather in Hangzhou.")
    assert msg.tool_calls


def test_live_native_json_structured_output() -> None:
    llm = _live_model(thinking={"type": "disabled"})
    chain = llm.with_structured_output(JsonAnswer, method="json_mode")
    parsed = chain.invoke("Return JSON with answer='ok'. Include the word json.")
    assert isinstance(parsed, JsonAnswer)


def test_live_reasoning_stream() -> None:
    llm = _live_model(thinking={"type": "enabled"}, reasoning_effort="high")
    chunks = list(llm.stream("9.11 and 9.8, which is greater?"))
    assert any(chunk.additional_kwargs.get("reasoning_content") for chunk in chunks)
    assert "".join(str(chunk.content) for chunk in chunks)


def test_live_multi_turn_conversation() -> None:
    llm = _live_model(thinking={"type": "disabled"})
    first = llm.invoke("Remember this exact token for the next turn: bluebird-json.")
    second = llm.invoke(
        [
            ("user", "Remember this exact token for the next turn: bluebird-json."),
            first,
            ("user", "What exact token did I ask you to remember?"),
        ]
    )
    assert "bluebird-json" in str(second.content)


def test_live_multi_turn_reasoning_stream() -> None:
    llm = _live_model(thinking={"type": "enabled"}, reasoning_effort="high")
    first = llm.invoke("9.11 and 9.8, which is greater? Answer briefly.")
    chunks = list(
        llm.stream(
            [
                ("user", "9.11 and 9.8, which is greater? Answer briefly."),
                first,
                ("user", "Now compare 3.14 and 3.2, briefly."),
            ]
        )
    )
    assert any(chunk.additional_kwargs.get("reasoning_content") for chunk in chunks)
    assert "".join(str(chunk.content) for chunk in chunks)
