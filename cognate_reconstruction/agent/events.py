"""Live, provider-neutral progress events for terminal and application harnesses."""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from typing import Any, Protocol, TextIO

from pydantic import Field

from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel


class AgentEventKind(StrEnum):
    NODE_START = "node_start"
    MODEL_TURN = "model_turn"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    NODE_COMMIT = "node_commit"
    NODE_COMPLETE = "node_complete"


class AgentEvent(WorkbenchModel):
    kind: AgentEventKind
    node_id: NonEmptyStr
    message: NonEmptyStr
    details: dict[str, Any] = Field(default_factory=dict)


class AgentEventSink(Protocol):
    def emit(self, event: AgentEvent) -> None: ...


class ConsoleEventSink:
    """Human-readable verbose trace written to stderr by default."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        max_json_chars: int = 4000,
    ) -> None:
        if max_json_chars < 200:
            raise ValueError("max_json_chars must be at least 200")
        self.stream = stream or sys.stderr
        self.max_json_chars = max_json_chars

    def emit(self, event: AgentEvent) -> None:
        prefix = f"[agent:{event.node_id}]"
        print(f"{prefix} {event.message}", file=self.stream, flush=True)
        if not event.details:
            return
        rendered = json.dumps(event.details, ensure_ascii=False, indent=2, sort_keys=True)
        if len(rendered) > self.max_json_chars:
            omitted = len(rendered) - self.max_json_chars
            rendered = rendered[: self.max_json_chars] + f"\n... ({omitted} characters omitted)"
        for line in rendered.splitlines():
            print(f"{prefix}   {line}", file=self.stream, flush=True)
