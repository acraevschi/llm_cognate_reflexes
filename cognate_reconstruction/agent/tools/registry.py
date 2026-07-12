"""Pydantic-backed native tool registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.schemas import (
    LLMToolCall,
    LLMToolDefinition,
    ToolError,
    ToolExecutionResult,
)
from cognate_reconstruction.schemas.common import WorkbenchModel

ToolHandler = Callable[[WorkbenchModel, AgentContext, str], WorkbenchModel]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[WorkbenchModel]
    handler: ToolHandler

    def definition(self) -> LLMToolDefinition:
        return LLMToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"tool {spec.name!r} is already registered")
        self._tools[spec.name] = spec

    def definitions(self) -> tuple[LLMToolDefinition, ...]:
        return tuple(self._tools[name].definition() for name in sorted(self._tools))

    def execute(self, call: LLMToolCall, context: AgentContext) -> ToolExecutionResult:
        spec = self._tools.get(call.name)
        if spec is None:
            return ToolExecutionResult(
                ok=False,
                error=ToolError(
                    error_type="unknown_tool",
                    message=f"unknown tool {call.name!r}",
                ),
            )
        try:
            # JSON validation permits JSON arrays for tuple fields while retaining
            # strict validation for scalar values.
            arguments = spec.args_model.model_validate_json(json.dumps(call.arguments))
            result = spec.handler(arguments, context, call.call_id)
        except (ValidationError, ValueError) as error:
            return ToolExecutionResult(
                ok=False,
                error=ToolError(
                    error_type=type(error).__name__,
                    message=str(error),
                ),
            )
        return ToolExecutionResult(ok=True, result=result.model_dump(mode="json"))
