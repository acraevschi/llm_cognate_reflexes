"""Provider abstraction used by the agent loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cognate_reconstruction.agent.schemas import LLMMessage, LLMToolDefinition


class LLMProvider(Protocol):
    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[LLMToolDefinition],
    ) -> LLMMessage: ...
