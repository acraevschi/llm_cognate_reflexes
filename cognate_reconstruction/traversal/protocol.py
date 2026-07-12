"""Protocol separating traversal from a concrete reconstruction algorithm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cognate_reconstruction.schemas.beam import NodeBeamState
from cognate_reconstruction.schemas.rules import AnomalyReport, ParsedSoundRule
from cognate_reconstruction.schemas.traversal import ReconstructionStep


class NodeReconstructor(Protocol):
    def reconstruct(
        self,
        parent_node_id: str,
        left: NodeBeamState,
        right: NodeBeamState,
        *,
        rules: Sequence[ParsedSoundRule] = (),
        anomalies: Sequence[AnomalyReport] = (),
    ) -> ReconstructionStep: ...
