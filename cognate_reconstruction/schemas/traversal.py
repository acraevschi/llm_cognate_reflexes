"""Serializable reconstruction-step and traversal state."""

from __future__ import annotations

from enum import StrEnum

from pydantic import model_validator

from cognate_reconstruction.schemas.alignment import CorrespondenceMap
from cognate_reconstruction.schemas.beam import NodeBeamState
from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel
from cognate_reconstruction.schemas.lexicon import ConceptMetadata, LanguageLexicon
from cognate_reconstruction.schemas.rules import AnomalyReport, RuleApplicationReport


class EvidenceKind(StrEnum):
    OBSERVED = "observed"
    RECONSTRUCTED = "reconstructed"


class EvidenceRelation(StrEnum):
    ACTIVE_CHILD = "active_child"
    DESCENDANT = "descendant"
    OUTGROUP = "outgroup"


class NodeEvidence(WorkbenchModel):
    node_id: NonEmptyStr
    kind: EvidenceKind
    relation: EvidenceRelation
    lexicon: LanguageLexicon
    descendant_leaf_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def validate_identity(self) -> NodeEvidence:
        if self.lexicon.variety_id != self.node_id:
            raise ValueError("evidence node and lexicon IDs must match")
        return self


class NodeReconstructionContext(WorkbenchModel):
    parent_node_id: NonEmptyStr
    active_child_ids: tuple[NonEmptyStr, ...]
    available_nodes: tuple[NodeEvidence, ...]
    concepts: tuple[ConceptMetadata, ...] = ()

    @model_validator(mode="after")
    def validate_context(self) -> NodeReconstructionContext:
        if len(self.active_child_ids) < 2:
            raise ValueError("reconstruction context requires at least two active children")
        if len(set(self.active_child_ids)) != len(self.active_child_ids):
            raise ValueError("active child IDs must be unique")
        node_ids = [node.node_id for node in self.available_nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("available evidence node IDs must be unique")
        concept_ids = [concept.concept_id for concept in self.concepts]
        if len(concept_ids) != len(set(concept_ids)):
            raise ValueError("context concept metadata IDs must be unique")
        return self


class ReconstructionStep(WorkbenchModel):
    parent_node_id: NonEmptyStr
    child_node_ids: tuple[NonEmptyStr, ...]
    input_beams: tuple[NodeBeamState, ...]
    correspondence_maps: tuple[CorrespondenceMap, ...] = ()
    output_beam: NodeBeamState
    rule_reports: tuple[RuleApplicationReport, ...] = ()
    anomaly_reports: tuple[AnomalyReport, ...] = ()


class TraversalSnapshot(WorkbenchModel):
    root_node_id: NonEmptyStr
    completed_node_ids: tuple[NonEmptyStr, ...]
    node_beams: tuple[NodeBeamState, ...]
    steps: tuple[ReconstructionStep, ...]
