"""High-level family inference results built on deterministic traversal."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cognate_reconstruction.agent.reconstructor import AgenticNodeReconstructor
from cognate_reconstruction.agent.trajectory import AgentTrajectory
from cognate_reconstruction.schemas.beam import NodeBeamState
from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel
from cognate_reconstruction.schemas.ingestion import IngestedDataset
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm
from cognate_reconstruction.schemas.traversal import TraversalSnapshot
from cognate_reconstruction.traversal.traverser import TreeTraverser


class InternalNodeVocabulary(WorkbenchModel):
    node_id: NonEmptyStr
    best_lexicon: LanguageLexicon
    beam: NodeBeamState


class FamilyReconstructionResult(WorkbenchModel):
    snapshot: TraversalSnapshot
    internal_nodes: tuple[InternalNodeVocabulary, ...]
    trajectories: tuple[AgentTrajectory, ...]


def _best_lexicon(beam: NodeBeamState) -> LanguageLexicon:
    return LanguageLexicon(
        variety_id=beam.node_id,
        name=beam.node_id,
        forms=tuple(
            LexicalForm(
                form_id=distribution.candidates[0].candidate_id,
                variety_id=beam.node_id,
                concept_id=distribution.concept_id,
                segments=distribution.candidates[0].segments,
            )
            for distribution in beam.distributions
        ),
    )


class ReconstructionService:
    """Reconstruct every internal vocabulary and return audit/training artifacts."""

    def __init__(
        self,
        reconstructor: AgenticNodeReconstructor,
    ) -> None:
        self.reconstructor = reconstructor
        self.beam_width = reconstructor.deterministic.beam_width

    def reconstruct_family(
        self,
        dataset: IngestedDataset,
        *,
        anchors_by_node: Mapping[str, Sequence[LexicalForm]] | None = None,
    ) -> FamilyReconstructionResult:
        self.reconstructor.clear_run_results()
        snapshot = TreeTraverser(
            beam_width=self.beam_width,
            reconstructor=self.reconstructor,
        ).traverse(dataset, anchors_by_node=anchors_by_node)
        internal_nodes = tuple(
            InternalNodeVocabulary(
                node_id=step.parent_node_id,
                best_lexicon=_best_lexicon(step.output_beam),
                beam=step.output_beam,
            )
            for step in snapshot.steps
        )
        return FamilyReconstructionResult(
            snapshot=snapshot,
            internal_nodes=internal_nodes,
            trajectories=tuple(
                result.trajectory for result in self.reconstructor.run_results
            ),
        )
