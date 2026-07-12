"""Protocol separating traversal from a concrete reconstruction algorithm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cognate_reconstruction.schemas.beam import NodeBeamState
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import (
    AnomalyReport,
    ParsedSoundRule,
    ReconstructionRule,
)
from cognate_reconstruction.schemas.traversal import (
    NodeReconstructionContext,
    ReconstructionStep,
)


class NodeReconstructor(Protocol):
    def reconstruct(
        self,
        parent_node_id: str,
        children: Sequence[NodeBeamState],
        *,
        rules: Sequence[ReconstructionRule | ParsedSoundRule] = (),
        anomalies: Sequence[AnomalyReport] = (),
        anchors: Sequence[LexicalForm] = (),
        evidence_context: NodeReconstructionContext | None = None,
    ) -> ReconstructionStep: ...
