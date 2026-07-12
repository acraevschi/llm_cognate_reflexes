"""LiteLLM adapter for OpenAI, Anthropic, Gemini, and open-weight models."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cognate_reconstruction.agent.schemas import (
    LLMMessage,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)


def _value(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


class LiteLLMProvider:
    """Normalize LiteLLM's OpenAI-shaped native tool-calling response."""

    def __init__(
        self,
        model: str,
        *,
        completion_kwargs: Mapping[str, Any] | None = None,
        completion_fn: Callable[..., object] | None = None,
    ) -> None:
        self.model = model
        self.completion_kwargs = dict(completion_kwargs or {})
        reserved = {"model", "messages", "tools", "tool_choice"}
        if overlap := sorted(reserved & self.completion_kwargs.keys()):
            raise ValueError(f"completion_kwargs contains reserved keys: {overlap}")
        self._completion_fn = completion_fn

    @staticmethod
    def _message_payload(message: LLMMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role.value}
        if message.content is not None:
            payload["content"] = message.content
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
            payload["name"] = message.name
        return payload

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[LLMToolDefinition],
    ) -> LLMMessage:
        completion = self._completion_fn
        if completion is None:
            try:
                from litellm import completion
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "LiteLLMProvider requires the optional 'agent' dependency"
                ) from error
        response = completion(
            model=self.model,
            messages=[self._message_payload(message) for message in messages],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            tool_choice="auto",
            **self.completion_kwargs,
        )
        choices = _value(response, "choices")
        if not choices:
            raise ValueError("LLM provider returned no choices")
        raw_message = _value(choices[0], "message")
        calls: list[LLMToolCall] = []
        for raw_call in _value(raw_message, "tool_calls", ()) or ():
            function = _value(raw_call, "function")
            call_id = _value(raw_call, "id")
            name = _value(function, "name")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("tool call is missing a non-empty ID")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool call is missing a non-empty function name")
            raw_arguments = _value(function, "arguments", "{}")
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else dict(raw_arguments)
            )
            if not isinstance(arguments, dict):
                raise ValueError("tool-call arguments must decode to a JSON object")
            calls.append(
                LLMToolCall(
                    call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )
        return LLMMessage(
            role=MessageRole.ASSISTANT,
            content=_value(raw_message, "content"),
            tool_calls=tuple(calls),
        )
