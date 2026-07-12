"""Serializable reconstruction-step and traversal state."""

from __future__ import annotations

from cognate_reconstruction.schemas.alignment import CorrespondenceMap
from cognate_reconstruction.schemas.beam import NodeBeamState
from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel
from cognate_reconstruction.schemas.rules import AnomalyReport, RuleApplicationReport


class ReconstructionStep(WorkbenchModel):
    parent_node_id: NonEmptyStr
    child_node_ids: tuple[NonEmptyStr, NonEmptyStr]
    input_beams: tuple[NodeBeamState, NodeBeamState]
    correspondence_maps: tuple[CorrespondenceMap, ...] = ()
    output_beam: NodeBeamState
    rule_reports: tuple[RuleApplicationReport, ...] = ()
    anomaly_reports: tuple[AnomalyReport, ...] = ()


class TraversalSnapshot(WorkbenchModel):
    root_node_id: NonEmptyStr
    completed_node_ids: tuple[NonEmptyStr, ...]
    node_beams: tuple[NodeBeamState, ...]
    steps: tuple[ReconstructionStep, ...]
