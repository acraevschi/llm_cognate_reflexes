"""Agentic NodeReconstructor wrapper keeping the deterministic core LLM-free."""

from __future__ import annotations

from collections.abc import Sequence

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.orchestrator import AgentOrchestrator
from cognate_reconstruction.agent.trajectory import AgentRunResult
from cognate_reconstruction.alignment.lingpy_adapter import LingPyAligner
from cognate_reconstruction.alignment.protocol import AlignmentProvider
from cognate_reconstruction.schemas.beam import NodeBeamState
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import (
    AnchorPolicy,
    AnomalyReport,
    ParsedSoundRule,
    ReconstructionRule,
)
from cognate_reconstruction.schemas.traversal import ReconstructionStep
from cognate_reconstruction.schemas.traversal import NodeReconstructionContext
from cognate_reconstruction.traversal.beam import beam_to_lexicon
from cognate_reconstruction.traversal.reconstructor import RuleBasedReconstructor


def _apply_overlay(
    beam: NodeBeamState,
    context: AgentContext,
    overlay_id: str | None,
) -> NodeBeamState:
    if overlay_id is None:
        return beam
    forms = context.forms_for_overlay(overlay_id)
    distributions = tuple(
        distribution.model_copy(
            update={
                "candidates": tuple(
                    candidate.model_copy(
                        update={"segments": forms[candidate.candidate_id].segments}
                    )
                    for candidate in distribution.candidates
                )
            }
        )
        for distribution in beam.distributions
    )
    return beam.model_copy(update={"distributions": distributions})


class AgenticNodeReconstructor:
    """Run one hypothesis-manager session, then invoke deterministic scoring."""

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        *,
        deterministic: RuleBasedReconstructor | None = None,
        aligner: AlignmentProvider | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.deterministic = deterministic or RuleBasedReconstructor()
        self.aligner = aligner or LingPyAligner()
        self.run_results: list[AgentRunResult] = []

    def clear_run_results(self) -> None:
        self.run_results.clear()

    def reconstruct(
        self,
        parent_node_id: str,
        children: Sequence[NodeBeamState],
        *,
        rules: Sequence[ReconstructionRule | ParsedSoundRule] = (),
        anomalies: Sequence[AnomalyReport] = (),
        anchors: Sequence[LexicalForm] = (),
        evidence_context: NodeReconstructionContext | None = None,
    ) -> ReconstructionStep:
        if rules or anomalies:
            raise ValueError(
                "AgenticNodeReconstructor does not accept precommitted rules or anomalies"
            )
        child_beams = tuple(children)
        context = AgentContext(
            node_id=parent_node_id,
            child_lexicons=tuple(beam_to_lexicon(child) for child in child_beams),
            aligner=self.aligner,
            anchors=(
                ()
                if self.deterministic.anchor_policy is AnchorPolicy.IGNORE
                else tuple(anchors)
            ),
            anchor_policy=self.deterministic.anchor_policy,
            evidence=evidence_context.available_nodes if evidence_context else (),
            concepts=evidence_context.concepts if evidence_context else (),
        )
        run_result = self.orchestrator.run(context)
        committed = run_result.reconstruction
        scored_children = tuple(
            _apply_overlay(
                child,
                context,
                committed.request.segmentation_overlay_id,
            )
            for child in child_beams
        )
        step = self.deterministic.reconstruct(
            parent_node_id,
            scored_children,
            rules=committed.parsed_rules,
            anomalies=committed.request.anomalies,
            anchors=anchors,
            evidence_context=evidence_context,
        )
        finalized = self.orchestrator.finalize(run_result, step)
        self.run_results.append(finalized)
        return step
