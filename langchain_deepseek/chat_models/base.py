"""DeepSeek chat model integration."""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from operator import itemgetter
from typing import Any, Literal, TypeAlias, cast

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_deepseek.chat_models._convert import (
    _convert_chunk_to_message_chunk,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from langchain_deepseek.chat_models._utils import (
    _create_usage_metadata,
    _dump_response,
    _is_pydantic_class,
    _normalize_payload_for_openai_sdk,
    _strip_internal_kwargs,
)
from langchain_deepseek.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_BETA_API_BASE = "https://api.deepseek.com/beta"

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatDeepSeek(BaseChatModel):
    """DeepSeek chat model integration for the current DeepSeek API.

    The implementation intentionally performs message conversion itself so
    DeepSeek-specific fields, especially `reasoning_content`, are preserved in
    non-streaming responses, streaming chunks, tool-call loops, and subsequent
    multi-turn requests.
    """

    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)
    root_client: Any = Field(default=None, exclude=True)
    root_async_client: Any = Field(default=None, exclude=True)

    model_name: str = Field(alias="model")
    """DeepSeek model name, e.g. `deepseek-v4-pro`."""

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("DEEPSEEK_API_KEY", default=None),
    )
    """DeepSeek API key."""

    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("DEEPSEEK_API_BASE", default=DEFAULT_API_BASE),
    )
    """DeepSeek OpenAI-compatible base URL."""

    request_timeout: float | None = Field(default=None, alias="timeout")
    max_retries: int = 2
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: list[str] | str | None = Field(default=None, alias="stop_sequences")
    n: int = Field(default=1, ge=1)
    streaming: bool = False
    stream_usage: bool = True
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    thinking: dict[str, Literal["enabled", "disabled"]] | None = None
    """Thinking mode switch, sent through OpenAI SDK `extra_body`."""

    reasoning_effort: Literal["high", "max", "low", "medium", "xhigh"] | None = None
    """DeepSeek reasoning effort. Compatibility values are accepted by the API."""

    logprobs: bool | None = None
    top_logprobs: int | None = None

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Move unknown constructor arguments into `model_kwargs`."""
        all_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_field_names:
                warnings.warn(
                    f"WARNING! {field_name} is not a default parameter. "
                    "It was transferred to model_kwargs.",
                    stacklevel=2,
                )
                extra[field_name] = values.pop(field_name)
        invalid = all_field_names.intersection(extra.keys())
        if invalid:
            msg = (
                f"Parameters {invalid} should be specified explicitly, not inside "
                "model_kwargs."
            )
            raise ValueError(msg)
        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate client parameters and construct OpenAI SDK clients."""
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, DEEPSEEK_API_KEY must be set."
            raise ValueError(msg)
        if self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params: dict[str, Any] = {
            "api_key": self.api_key.get_secret_value() if self.api_key else None,
            "base_url": self.api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        client_params = {k: v for k, v in client_params.items() if v is not None}

        if self.client is None:
            self.root_client = openai.OpenAI(**client_params)
            self.client = self.root_client.chat.completions
        if self.async_client is None:
            self.root_async_client = openai.AsyncOpenAI(**client_params)
            self.async_client = self.root_async_client.chat.completions
        return self

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self.model_name

    @property
    def _llm_type(self) -> str:
        return "chat-deepseek"

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "DEEPSEEK_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "streaming": self.streaming,
            "thinking": self.thinking,
            "reasoning_effort": self.reasoning_effort,
            "model_kwargs": self.model_kwargs,
        }

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="deepseek",
            ls_model_name=params.get("model", self.model_name),
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop if isinstance(ls_stop, list) else [ls_stop]
        return ls_params

    @property
    def _default_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model_name, **self.model_kwargs}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            params["stop"] = self.stop
        if self.n > 1:
            params["n"] = self.n
        if self.logprobs is not None:
            params["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            params["top_logprobs"] = self.top_logprobs
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.thinking is not None:
            extra_body = dict(params.get("extra_body") or {})
            extra_body["thinking"] = self.thinking
            params["extra_body"] = extra_body
        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _normalize_payload_for_openai_sdk(params)
        return {"messages": message_dicts, **params}

    def _create_message_dicts(
        self, messages: list[BaseMessage], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        params = self._default_params
        if stop is not None:
            params["stop"] = stop
        return [_convert_message_to_dict(m) for m in messages], params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return generate_from_stream(
                self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _strip_internal_kwargs(params)
        _normalize_payload_for_openai_sdk(params)
        response = self.client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return await agenerate_from_stream(
                self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        _strip_internal_kwargs(params)
        _normalize_payload_for_openai_sdk(params)
        response = await self.async_client.create(messages=message_dicts, **params)
        return self._create_chat_result(response)

    def _stream(  # noqa: C901
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage:
            params["stream_options"] = {"include_usage": True}
        _strip_internal_kwargs(params)
        _normalize_payload_for_openai_sdk(params)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        for chunk in self.client.create(messages=message_dicts, **params):
            chunk_dict = _dump_response(chunk)
            generation_chunk = self._convert_chunk_to_generation_chunk(
                chunk_dict, default_chunk_class
            )
            if generation_chunk is None:
                continue
            default_chunk_class = generation_chunk.message.__class__
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=(generation_chunk.generation_info or {}).get("logprobs"),
                )
            yield generation_chunk

    async def _astream(  # noqa: C901
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}
        if self.stream_usage:
            params["stream_options"] = {"include_usage": True}
        _strip_internal_kwargs(params)
        _normalize_payload_for_openai_sdk(params)

        default_chunk_class: type[BaseMessageChunk] = AIMessageChunk
        async for chunk in await self.async_client.create(
            messages=message_dicts, **params
        ):
            chunk_dict = _dump_response(chunk)
            generation_chunk = self._convert_chunk_to_generation_chunk(
                chunk_dict, default_chunk_class
            )
            if generation_chunk is None:
                continue
            default_chunk_class = generation_chunk.message.__class__
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=generation_chunk.text,
                    chunk=generation_chunk,
                    logprobs=(generation_chunk.generation_info or {}).get("logprobs"),
                )
            yield generation_chunk

    def _create_chat_result(self, response: Any) -> ChatResult:
        response_dict = _dump_response(response)
        if error := response_dict.get("error"):
            msg = (
                f"DeepSeek API returned an error: {error.get('message', str(error))} "
                f"(code: {error.get('code', 'unknown')})"
            )
            raise ValueError(msg)

        generations: list[ChatGeneration] = []
        token_usage = response_dict.get("usage") or {}
        response_model = response_dict.get("model")
        system_fingerprint = response_dict.get("system_fingerprint")

        for choice in response_dict.get("choices", []):
            message = _convert_dict_to_message(choice["message"])
            if isinstance(message, AIMessage):
                if token_usage:
                    message.usage_metadata = _create_usage_metadata(token_usage)
                message.response_metadata.update(
                    {
                        "model_provider": "deepseek",
                        "model_name": response_model or self.model_name,
                    }
                )
                if system_fingerprint:
                    message.response_metadata["system_fingerprint"] = system_fingerprint
            generation_info = {
                "finish_reason": choice.get("finish_reason"),
                "model_name": response_model or self.model_name,
            }
            if "logprobs" in choice:
                generation_info["logprobs"] = choice["logprobs"]
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )

        llm_output = {"model_name": response_model or self.model_name}
        for key in ("id", "object", "created"):
            if key in response_dict:
                llm_output[key] = response_dict[key]
        return ChatResult(generations=generations, llm_output=llm_output)

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: Mapping[str, Any],
        default_chunk_class: type[BaseMessageChunk],
        _base_generation_info: dict[str, Any] | None = None,
    ) -> ChatGenerationChunk | None:
        if not chunk.get("choices"):
            if usage := chunk.get("usage"):
                message_chunk = AIMessageChunk(
                    content="",
                    usage_metadata=_create_usage_metadata(usage),
                    response_metadata={"model_provider": "deepseek"},
                )
                return ChatGenerationChunk(message=message_chunk)
            return None

        choice = chunk["choices"][0]
        message_chunk = _convert_chunk_to_message_chunk(chunk, default_chunk_class)
        generation_info: dict[str, Any] = {}
        if finish_reason := choice.get("finish_reason"):
            generation_info["finish_reason"] = finish_reason
            generation_info["model_name"] = chunk.get("model") or self.model_name
            for key in ("id", "created", "object", "system_fingerprint"):
                if key in chunk:
                    generation_info[key] = chunk[key]
        if logprobs := choice.get("logprobs"):
            generation_info["logprobs"] = logprobs
        if generation_info and isinstance(message_chunk, AIMessageChunk):
            message_chunk = message_chunk.model_copy(
                update={
                    "response_metadata": {
                        **message_chunk.response_metadata,
                        **generation_info,
                        "model_provider": "deepseek",
                    }
                }
            )
        return ChatGenerationChunk(
            message=message_chunk,
            generation_info=generation_info or None,
        )

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind OpenAI-compatible tools to the DeepSeek model."""
        formatted_tools = [
            convert_to_openai_tool(tool, strict=strict) for tool in tools
        ]
        if tool_choice is not None and tool_choice:
            if tool_choice == "any":
                tool_choice = "required"
            if isinstance(tool_choice, str) and (
                tool_choice not in ("auto", "none", "required")
            ):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(formatted_tools) != 1:
                    msg = "tool_choice=True requires exactly one tool."
                    raise ValueError(msg)
                tool_choice = {
                    "type": "function",
                    "function": {"name": formatted_tools[0]["function"]["name"]},
                }
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(  # type: ignore[override]
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_mode",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Return a wrapper that parses model output to a schema.

        DeepSeek's native JSON Output currently exposes `response_format={
        "type": "json_object" }`, not OpenAI's `json_schema` shape. Therefore
        `json_mode` and `json_schema` both use DeepSeek JSON Output plus a
        LangChain parser. Use `function_calling` for tool-based structured
        output.
        """
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                msg = "schema must be specified for function_calling."
                raise ValueError(msg)
            formatted_tool = convert_to_openai_tool(schema)
            tool_name = formatted_tool["function"]["name"]
            llm = self.bind_tools(
                [schema],
                tool_choice=tool_name,
                strict=strict,
                ls_structured_output_format={
                    "kwargs": {"method": "function_calling", "strict": strict},
                    "schema": formatted_tool,
                },
                **kwargs,
            )
            output_parser: OutputParserLike = (
                PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
                if is_pydantic_schema
                else JsonOutputKeyToolsParser(key_name=tool_name, first_tool_only=True)
            )
        elif method in {"json_mode", "json_schema"}:
            response_format = {"type": "json_object"}
            ls_format_info: dict[str, Any] = {
                "kwargs": {"method": method, "strict": strict},
            }
            if schema is not None:
                ls_format_info["schema"] = convert_to_json_schema(schema)
            llm = self.bind(
                response_format=response_format,
                ls_structured_output_format=ls_format_info,
                **kwargs,
            )
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            msg = (
                "Unrecognized method argument. Expected 'function_calling', "
                f"'json_mode', or 'json_schema'. Received: {method!r}"
            )
            raise ValueError(msg)

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser,
                parsing_error=lambda _: None,
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser
