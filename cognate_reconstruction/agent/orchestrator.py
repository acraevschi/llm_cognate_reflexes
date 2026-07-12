"""Bounded native-tool-calling loop for one internal reconstruction node."""

from __future__ import annotations

import hashlib
import json
import uuid

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.events import (
    AgentEvent,
    AgentEventKind,
    AgentEventSink,
)
from cognate_reconstruction.agent.instructions import load_agent_instructions
from cognate_reconstruction.agent.providers.protocol import LLMProvider
from cognate_reconstruction.agent.schemas import (
    CommittedReconstruction,
    LLMMessage,
    MessageRole,
    NodeLexiconSummary,
    NodePromptPayload,
)
from cognate_reconstruction.agent.tools import ToolRegistry, default_tool_registry
from cognate_reconstruction.agent.trajectory import (
    AgentRunResult,
    AgentTrajectory,
    TrajectorySink,
)
from cognate_reconstruction.schemas.traversal import ReconstructionStep


class AgentLoopLimitError(RuntimeError):
    pass


class AgentOrchestrator:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        registry: ToolRegistry | None = None,
        max_turns: int = 24,
        max_tool_calls: int = 64,
        instructions: str | None = None,
        trajectory_sink: TrajectorySink | None = None,
        event_sink: AgentEventSink | None = None,
    ) -> None:
        if max_turns < 1 or max_tool_calls < 1:
            raise ValueError("agent loop limits must be positive")
        self.provider = provider
        self.registry = registry or default_tool_registry()
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.instructions = instructions or load_agent_instructions()
        self.trajectory_sink = trajectory_sink
        self.event_sink = event_sink

    def _emit(
        self,
        kind: AgentEventKind,
        node_id: str,
        message: str,
        **details: object,
    ) -> None:
        if self.event_sink is not None:
            self.event_sink.emit(
                AgentEvent(
                    kind=kind,
                    node_id=node_id,
                    message=message,
                    details=dict(details),
                )
            )

    def _trajectory(
        self,
        context: AgentContext,
        payload: NodePromptPayload,
        messages: list[LLMMessage],
        *,
        completed: bool,
        failure: str | None = None,
        write_to_sink: bool = True,
    ) -> AgentTrajectory:
        definitions = self.registry.definitions()
        trajectory = AgentTrajectory(
            trajectory_id=f"trajectory-{uuid.uuid4()}",
            node_id=context.node_id,
            provider_adapter=type(self.provider).__name__,
            model_id=getattr(self.provider, "model", None),
            instruction_sha256=hashlib.sha256(self.instructions.encode()).hexdigest(),
            tool_schema_sha256=hashlib.sha256(
                json.dumps(
                    [definition.model_dump(mode="json") for definition in definitions],
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            initial_payload=payload,
            tool_definitions=definitions,
            messages=tuple(messages),
            committed_reconstruction=context.commit,
            completed=completed,
            failure=failure,
        )
        if write_to_sink and self.trajectory_sink is not None:
            self.trajectory_sink.write(trajectory)
        return trajectory

    def finalize(
        self,
        run_result: AgentRunResult,
        step: ReconstructionStep,
    ) -> AgentRunResult:
        """Attach deterministic outcome data and emit the completed trajectory."""
        trajectory = run_result.trajectory.model_copy(
            update={"reconstruction_step": step}
        )
        if self.trajectory_sink is not None:
            self.trajectory_sink.write(trajectory)
        self._emit(
            AgentEventKind.NODE_COMPLETE,
            step.parent_node_id,
            f"reconstructed {len(step.output_beam.distributions)} concepts",
            child_node_ids=list(step.child_node_ids),
            output_candidates=sum(
                len(distribution.candidates)
                for distribution in step.output_beam.distributions
            ),
        )
        return AgentRunResult(
            reconstruction=run_result.reconstruction,
            trajectory=trajectory,
        )

    def run(self, context: AgentContext) -> AgentRunResult:
        payload = NodePromptPayload(
            node_id=context.node_id,
            active_children=tuple(
                NodeLexiconSummary(
                    node_id=lexicon.variety_id,
                    name=lexicon.name,
                    form_count=len(lexicon.forms),
                    concept_count=len({form.concept_id for form in lexicon.forms}),
                )
                for lexicon in context.child_lexicons
            ),
            anchor_policy=context.anchor_policy,
            anchors=context.anchors,
        )
        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=self.instructions),
            LLMMessage(
                role=MessageRole.USER,
                content=(
                    "Reconstruct the parent represented by this node. Use the tools "
                    "iteratively and finish with commit_reconstruction.\n\n"
                    + payload.model_dump_json(indent=2)
                ),
            ),
        ]
        self._emit(
            AgentEventKind.NODE_START,
            context.node_id,
            "starting reconstruction",
            active_child_ids=list(context.child_ids),
            anchor_policy=context.anchor_policy.value,
            available_evidence_nodes=len(context.evidence),
        )
        tool_calls_used = 0
        for turn_index in range(1, self.max_turns + 1):
            self._emit(
                AgentEventKind.MODEL_TURN,
                context.node_id,
                f"requesting model turn {turn_index}",
                message_count=len(messages),
                tool_count=len(self.registry.definitions()),
            )
            reply = self.provider.complete(messages, self.registry.definitions())
            if reply.role is not MessageRole.ASSISTANT:
                raise ValueError("LLM providers must return an assistant message")
            messages.append(reply)
            self._emit(
                AgentEventKind.MODEL_RESPONSE,
                context.node_id,
                f"model returned {len(reply.tool_calls)} tool call(s)",
                content=reply.content,
                tool_names=[call.name for call in reply.tool_calls],
            )
            if not reply.tool_calls:
                messages.append(
                    LLMMessage(
                        role=MessageRole.USER,
                        content=(
                            "Continue by calling an available tool. The session ends only "
                            "after a valid commit_reconstruction call."
                        ),
                    )
                )
                continue
            for call in reply.tool_calls:
                tool_calls_used += 1
                if tool_calls_used > self.max_tool_calls:
                    message = "agent exceeded its tool-call limit"
                    self._trajectory(
                        context,
                        payload,
                        messages,
                        completed=False,
                        failure=message,
                    )
                    raise AgentLoopLimitError(message)
                self._emit(
                    AgentEventKind.TOOL_CALL,
                    context.node_id,
                    f"calling tool {call.name}",
                    call_id=call.call_id,
                    arguments=call.arguments,
                )
                result = self.registry.execute(call, context)
                self._emit(
                    AgentEventKind.TOOL_RESULT,
                    context.node_id,
                    f"tool {call.name} {'succeeded' if result.ok else 'failed'}",
                    call_id=call.call_id,
                    result=result.model_dump(mode="json"),
                )
                messages.append(
                    LLMMessage(
                        role=MessageRole.TOOL,
                        content=result.model_dump_json(),
                        tool_call_id=call.call_id,
                        name=call.name,
                    )
                )
                if context.commit is not None:
                    self._emit(
                        AgentEventKind.NODE_COMMIT,
                        context.node_id,
                        "accepted reconstruction commit",
                        rule_ids=[
                            rule.rule.rule_id for rule in context.commit.parsed_rules
                        ],
                        anomaly_count=len(context.commit.request.anomalies),
                    )
                    trajectory = self._trajectory(
                        context,
                        payload,
                        messages,
                        completed=True,
                        write_to_sink=False,
                    )
                    return AgentRunResult(
                        reconstruction=context.commit,
                        trajectory=trajectory,
                    )
        message = "agent did not commit within its turn limit"
        self._trajectory(
            context,
            payload,
            messages,
            completed=False,
            failure=message,
        )
        raise AgentLoopLimitError(message)
