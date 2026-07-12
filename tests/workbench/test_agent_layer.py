from __future__ import annotations

from collections.abc import Sequence
from io import StringIO
import re

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.orchestrator import AgentOrchestrator
from cognate_reconstruction.agent.events import ConsoleEventSink
from cognate_reconstruction.agent.providers import LiteLLMProvider
from cognate_reconstruction.agent.reconstructor import AgenticNodeReconstructor
from cognate_reconstruction.agent.service import ReconstructionService
from cognate_reconstruction.agent.schemas import (
    LLMMessage,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)
from cognate_reconstruction.agent.tools import default_tool_registry
from cognate_reconstruction.agent.trajectory import (
    JsonlTrajectorySink,
    TrajectoryDatasetBuilder,
)
from cognate_reconstruction.alignment.lingpy_adapter import LingPyAligner
from cognate_reconstruction.ingestion import ingest_payload
from cognate_reconstruction.schemas.ingestion import WorkbenchPayload
from cognate_reconstruction.schemas.lexicon import (
    ConceptMetadata,
    LanguageLexicon,
    LexicalForm,
)
from cognate_reconstruction.schemas.traversal import (
    EvidenceKind,
    EvidenceRelation,
    NodeEvidence,
)


def lexicon(variety_id: str, segment: str) -> LanguageLexicon:
    return LanguageLexicon(
        variety_id=variety_id,
        name=variety_id,
        forms=(
            LexicalForm(
                form_id=f"{variety_id}:water",
                variety_id=variety_id,
                concept_id="water",
                segments=(segment,),
            ),
        ),
    )


def context() -> AgentContext:
    return AgentContext(
        node_id="PROTO",
        child_lexicons=(lexicon("A", "p"), lexicon("B", "p"), lexicon("C", "f")),
        anchors=(),
        # Alignment is not invoked in these focused registry tests.
        aligner=LingPyAligner(),
    )


def test_invalid_dsl_is_returned_as_tool_error() -> None:
    result = default_tool_registry().execute(
        LLMToolCall(
            call_id="bad-rule",
            name="test_sound_law",
            arguments={"dsl": "p f", "source_child_ids": ["A"]},
        ),
        context(),
    )
    assert not result.ok
    assert result.error is not None
    assert "exactly one '>'" in result.error.message


def test_commit_requires_and_reuses_exact_validation() -> None:
    state = context()
    registry = default_tool_registry()
    validation = registry.execute(
        LLMToolCall(
            call_id="validate-p",
            name="test_sound_law",
            arguments={
                "dsl": "p > f",
                "source_child_ids": ["A", "B"],
            },
        ),
        state,
    )
    assert validation.ok
    commit = registry.execute(
        LLMToolCall(
            call_id="commit",
            name="commit_reconstruction",
            arguments={
                "node_id": "PROTO",
                "rules": [
                    {
                        "rule_id": "frication",
                        "dsl": "p > f",
                        "source_child_ids": ["A", "B"],
                        "confidence": 0.8,
                        "validation_call_id": "validate-p",
                        "supporting_form_ids": ["A:water", "B:water"],
                        "rationale": "regular correspondence in both children",
                    }
                ],
                "anomalies": [],
                "summary": "Reconstruct p as f for the two supporting branches.",
            },
        ),
        state,
    )
    assert commit.ok
    assert state.commit is not None
    assert state.commit.parsed_rules[0].source_child_ids == ("A", "B")


def test_committed_segmentation_overlay_matches_rule_validation() -> None:
    state = context()
    registry = default_tool_registry()
    segmented = registry.execute(
        LLMToolCall(
            call_id="segment",
            name="segment_morphemes",
            arguments={
                "segmentations": [{"form_id": "A:water", "segments": ["p", "+"]}],
                "rationale": "mark a root boundary",
            },
        ),
        state,
    )
    assert segmented.ok and segmented.result is not None
    overlay_id = segmented.result["segmentation_overlay_id"]
    tested = registry.execute(
        LLMToolCall(
            call_id="validate-boundary",
            name="test_sound_law",
            arguments={
                "dsl": "p > f / _ +",
                "source_child_ids": ["A"],
                "segmentation_overlay_id": overlay_id,
            },
        ),
        state,
    )
    assert tested.ok
    committed = registry.execute(
        LLMToolCall(
            call_id="commit-boundary",
            name="commit_reconstruction",
            arguments={
                "node_id": "PROTO",
                "segmentation_overlay_id": overlay_id,
                "rules": [
                    {
                        "rule_id": "root-frication",
                        "dsl": "p > f / _ +",
                        "source_child_ids": ["A"],
                        "confidence": 0.8,
                        "validation_call_id": "validate-boundary",
                        "supporting_form_ids": ["A:water"],
                        "rationale": "conditioned at the marked boundary",
                    }
                ],
                "anomalies": [],
                "summary": "Boundary-conditioned reconstruction.",
            },
        ),
        state,
    )
    assert committed.ok


class ScriptedProvider:
    def __init__(self) -> None:
        self.turn = 0

    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[LLMToolDefinition],
    ) -> LLMMessage:
        assert tools
        self.turn += 1
        if self.turn == 1:
            return LLMMessage(
                role=MessageRole.ASSISTANT,
                tool_calls=(
                    LLMToolCall(
                        call_id="validate",
                        name="test_sound_law",
                        arguments={"dsl": "p > f", "source_child_ids": ["A", "B"]},
                    ),
                ),
            )
        assert messages[-1].role is MessageRole.TOOL
        return LLMMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(
                LLMToolCall(
                    call_id="commit",
                    name="commit_reconstruction",
                    arguments={
                        "node_id": "PROTO",
                        "rules": [
                            {
                                "rule_id": "frication",
                                "dsl": "p > f",
                                "source_child_ids": ["A", "B"],
                                "confidence": 0.75,
                                "validation_call_id": "validate",
                                "supporting_form_ids": ["A:water", "B:water"],
                                "rationale": "tested on both p-reflexes",
                            }
                        ],
                        "anomalies": [],
                        "summary": "Validated operational reconstruction.",
                    },
                ),
            ),
        )


def test_orchestrator_loops_until_commit() -> None:
    run_result = AgentOrchestrator(
        ScriptedProvider(), instructions="Use tools, then commit."
    ).run(context())
    committed = run_result.reconstruction
    assert committed.request.node_id == "PROTO"
    assert committed.parsed_rules[0].confidence == 0.75
    assert run_result.trajectory.completed
    assert run_result.trajectory.committed_reconstruction == committed


def test_litellm_provider_normalizes_native_tool_calls() -> None:
    captured = {}

    def completion(**kwargs: object) -> object:
        captured.update(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "test_sound_law",
                                    "arguments": '{"dsl":"p > f","source_child_ids":["A"]}',
                                },
                            }
                        ],
                    }
                }
            ]
        }

    provider = LiteLLMProvider("openai/test-model", completion_fn=completion)
    reply = provider.complete(
        (LLMMessage(role=MessageRole.USER, content="test"),),
        default_tool_registry().definitions(),
    )
    assert captured["tool_choice"] == "auto"
    assert reply.tool_calls[0].name == "test_sound_law"
    assert reply.tool_calls[0].arguments["source_child_ids"] == ["A"]


def test_agent_can_search_semantics_segments_and_available_tree_nodes() -> None:
    active_a = lexicon("A", "p")
    active_b = lexicon("B", "b")
    outgroup = lexicon("OUT", "n")
    state = AgentContext(
        node_id="PROTO",
        child_lexicons=(active_a, active_b),
        aligner=LingPyAligner(),
        evidence=(
            NodeEvidence(
                node_id="A",
                kind=EvidenceKind.OBSERVED,
                relation=EvidenceRelation.ACTIVE_CHILD,
                lexicon=active_a,
                descendant_leaf_ids=("A",),
            ),
            NodeEvidence(
                node_id="B",
                kind=EvidenceKind.OBSERVED,
                relation=EvidenceRelation.ACTIVE_CHILD,
                lexicon=active_b,
                descendant_leaf_ids=("B",),
            ),
            NodeEvidence(
                node_id="OUT",
                kind=EvidenceKind.OBSERVED,
                relation=EvidenceRelation.OUTGROUP,
                lexicon=outgroup,
                descendant_leaf_ids=("OUT",),
            ),
        ),
        concepts=(ConceptMetadata(concept_id="water", gloss="drinking water"),),
    )
    result = default_tool_registry().execute(
        LLMToolCall(
            call_id="search",
            name="search_forms",
            arguments={
                "scope": "available_tree",
                "node_ids": ["OUT"],
                "concept_query": "drinking",
                "segment_pattern": ["n"],
                "position": "initial",
            },
        ),
        state,
    )
    assert result.ok and result.result is not None
    assert result.result["hits"][0]["node_id"] == "OUT"
    available = default_tool_registry().execute(
        LLMToolCall(
            call_id="nodes",
            name="list_available_nodes",
            arguments={"relations": ["outgroup"]},
        ),
        state,
    )
    assert available.ok and available.result is not None
    assert available.result["nodes"][0]["node_id"] == "OUT"
    assert "lexicon" not in available.result["nodes"][0]


class AutoCommitProvider:
    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[LLMToolDefinition],
    ) -> LLMMessage:
        assert tools
        match = re.search(r'"node_id":\s*"([^"]+)"', messages[1].content or "")
        assert match is not None
        node_id = match.group(1)
        return LLMMessage(
            role=MessageRole.ASSISTANT,
            tool_calls=(
                LLMToolCall(
                    call_id=f"commit:{node_id}",
                    name="commit_reconstruction",
                    arguments={
                        "node_id": node_id,
                        "rules": [],
                        "anomalies": [],
                        "summary": "Identity reconstruction is best supported.",
                    },
                ),
            ),
        )


def test_console_event_sink_lists_model_and_tool_actions() -> None:
    stream = StringIO()
    AgentOrchestrator(
        AutoCommitProvider(),
        instructions="Commit the deterministic result.",
        event_sink=ConsoleEventSink(stream=stream),
    ).run(context())
    rendered = stream.getvalue()
    assert "starting reconstruction" in rendered
    assert "requesting model turn 1" in rendered
    assert "calling tool commit_reconstruction" in rendered
    assert '"node_id": "PROTO"' in rendered
    assert "accepted reconstruction commit" in rendered


def test_family_service_returns_internal_vocabularies_and_training_trajectories(
    tmp_path,
) -> None:
    sink_path = tmp_path / "trajectories.jsonl"
    orchestrator = AgentOrchestrator(
        AutoCommitProvider(),
        instructions="Use deterministic tools and commit.",
        trajectory_sink=JsonlTrajectorySink(sink_path),
    )
    service = ReconstructionService(
        AgenticNodeReconstructor(orchestrator),
    )
    dataset = ingest_payload(
        WorkbenchPayload(
            lexicons=(lexicon("A", "p"), lexicon("B", "p"), lexicon("C", "p")),
            newick="((A,B)X,C)ROOT;",
        )
    )
    result = service.reconstruct_family(dataset)
    assert [item.node_id for item in result.internal_nodes] == ["X", "ROOT"]
    assert all(item.best_lexicon.forms for item in result.internal_nodes)
    assert len(result.trajectories) == 2
    loaded = TrajectoryDatasetBuilder.read_jsonl(sink_path)
    assert all(item.reconstruction_step is not None for item in loaded)
    examples = TrajectoryDatasetBuilder().build(loaded)
    assert len(examples) == 2
    assert any(
        call.name == "commit_reconstruction"
        for message in examples[0].messages
        for call in message.tool_calls
    )
