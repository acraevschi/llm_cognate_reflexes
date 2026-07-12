"""Versioned trajectory artifacts and backend-neutral training preparation."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import model_validator

from cognate_reconstruction.agent.schemas import (
    CommittedReconstruction,
    LLMMessage,
    LLMToolDefinition,
    NodePromptPayload,
)
from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel
from cognate_reconstruction.schemas.traversal import ReconstructionStep

TRAJECTORY_SCHEMA_VERSION = "1.0"


class AgentTrajectory(WorkbenchModel):
    trajectory_id: NonEmptyStr
    schema_version: Literal["1.0"] = TRAJECTORY_SCHEMA_VERSION
    node_id: NonEmptyStr
    provider_adapter: NonEmptyStr
    model_id: NonEmptyStr | None = None
    instruction_sha256: NonEmptyStr
    tool_schema_sha256: NonEmptyStr
    initial_payload: NodePromptPayload
    tool_definitions: tuple[LLMToolDefinition, ...]
    messages: tuple[LLMMessage, ...]
    committed_reconstruction: CommittedReconstruction | None = None
    reconstruction_step: ReconstructionStep | None = None
    completed: bool
    failure: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> AgentTrajectory:
        if self.completed and self.committed_reconstruction is None:
            raise ValueError("completed trajectories require a committed reconstruction")
        if not self.completed and self.failure is None:
            raise ValueError("incomplete trajectories require a failure explanation")
        if not self.completed and self.reconstruction_step is not None:
            raise ValueError("incomplete trajectories cannot contain a reconstruction step")
        return self


class AgentRunResult(WorkbenchModel):
    reconstruction: CommittedReconstruction
    trajectory: AgentTrajectory


class TrajectorySink(Protocol):
    def write(self, trajectory: AgentTrajectory) -> None: ...


class JsonlTrajectorySink:
    """Append immutable trajectory records without retaining a family run in RAM."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, trajectory: AgentTrajectory) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(trajectory.model_dump_json())
            handle.write("\n")


class TrainingExample(WorkbenchModel):
    example_id: NonEmptyStr
    schema_version: Literal["1.0"] = TRAJECTORY_SCHEMA_VERSION
    node_id: NonEmptyStr
    messages: tuple[LLMMessage, ...]
    tool_definitions: tuple[LLMToolDefinition, ...]
    reconstruction_step: ReconstructionStep | None = None
    source_trajectory_id: NonEmptyStr


class TrajectoryDatasetBuilder:
    """Create generic chat/tool examples consumable by later TRL/Unsloth adapters."""

    def build(
        self,
        trajectories: Sequence[AgentTrajectory],
        *,
        include_incomplete: bool = False,
    ) -> tuple[TrainingExample, ...]:
        return tuple(
            TrainingExample(
                example_id=f"example:{trajectory.trajectory_id}",
                node_id=trajectory.node_id,
                messages=trajectory.messages,
                tool_definitions=trajectory.tool_definitions,
                reconstruction_step=trajectory.reconstruction_step,
                source_trajectory_id=trajectory.trajectory_id,
            )
            for trajectory in trajectories
            if trajectory.completed or include_incomplete
        )

    @staticmethod
    def read_jsonl(path: str | Path) -> tuple[AgentTrajectory, ...]:
        with Path(path).open(encoding="utf-8") as handle:
            return tuple(
                AgentTrajectory.model_validate_json(line)
                for line in handle
                if line.strip()
            )

    @staticmethod
    def write_jsonl(
        examples: Iterable[TrainingExample],
        path: str | Path,
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example.model_dump(mode="json"), sort_keys=True))
                handle.write("\n")
