"""Deterministic rule-driven combination of two child beams."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import product

from cognate_reconstruction.rules.engine import RuleEngine
from cognate_reconstruction.schemas.beam import CandidateDerivation, NodeBeamState
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import (
    AnomalyReport,
    ParsedSoundRule,
    RuleApplicationReport,
)
from cognate_reconstruction.schemas.traversal import ReconstructionStep
from cognate_reconstruction.traversal.beam import RawCandidate, normalize_and_prune


class RuleBasedReconstructor:
    """Apply a rule cascade to child candidates and combine their probability mass.

    Rules are interpreted in their written direction. If historical inference
    requires inverse rules, callers must supply those inverse hypotheses
    explicitly; the engine never guesses an inverse for a non-bijective law.
    """

    def __init__(self, *, beam_width: int = 5, engine: RuleEngine | None = None) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        self.beam_width = beam_width
        self.engine = engine or RuleEngine()

    def _transform(
        self,
        parent_node_id: str,
        concept_id: str,
        candidate_id: str,
        segments: tuple[str, ...],
        rules: Sequence[ParsedSoundRule],
    ) -> tuple[tuple[str, ...], tuple[RuleApplicationReport, ...]]:
        form = LexicalForm(
            form_id=f"beam-form:{candidate_id}",
            variety_id=parent_node_id,
            concept_id=concept_id,
            segments=segments,
        )
        transformed, reports = self.engine.apply_rules(
            rules,
            (form,),
            source_candidate_ids={form.form_id: candidate_id},
        )
        return transformed[0].segments, reports

    def reconstruct(
        self,
        parent_node_id: str,
        left: NodeBeamState,
        right: NodeBeamState,
        *,
        rules: Sequence[ParsedSoundRule] = (),
        anomalies: Sequence[AnomalyReport] = (),
    ) -> ReconstructionStep:
        left_by_concept = {d.concept_id: d for d in left.distributions}
        right_by_concept = {d.concept_id: d for d in right.distributions}
        output_distributions = []
        all_reports: list[RuleApplicationReport] = []
        for concept_id in sorted(left_by_concept.keys() | right_by_concept.keys()):
            left_distribution = left_by_concept.get(concept_id)
            right_distribution = right_by_concept.get(concept_id)
            raw: list[RawCandidate] = []
            if left_distribution is not None and right_distribution is not None:
                for left_candidate, right_candidate in product(
                    left_distribution.candidates, right_distribution.candidates
                ):
                    joint_score = left_candidate.log_score + right_candidate.log_score
                    left_output, left_reports = self._transform(
                        parent_node_id,
                        concept_id,
                        left_candidate.candidate_id,
                        left_candidate.segments,
                        rules,
                    )
                    right_output, right_reports = self._transform(
                        parent_node_id,
                        concept_id,
                        right_candidate.candidate_id,
                        right_candidate.segments,
                        rules,
                    )
                    all_reports.extend(left_reports)
                    all_reports.extend(right_reports)
                    outputs = (left_output,) if left_output == right_output else (left_output, right_output)
                    branch_penalty = -math.log(len(outputs))
                    for output in outputs:
                        raw.append(
                            (
                                output,
                                joint_score + branch_penalty,
                                CandidateDerivation(
                                    derivation_id=(
                                        f"{parent_node_id}:{concept_id}:"
                                        f"{left_candidate.candidate_id}:{right_candidate.candidate_id}"
                                    ),
                                    child_candidate_ids=(
                                        left_candidate.candidate_id,
                                        right_candidate.candidate_id,
                                    ),
                                    rule_ids=tuple(rule.rule_id for rule in rules),
                                    note="joint child-beam derivation",
                                ),
                            )
                        )
            else:
                available = left_distribution or right_distribution
                assert available is not None
                for candidate in available.candidates:
                    output, reports = self._transform(
                        parent_node_id,
                        concept_id,
                        candidate.candidate_id,
                        candidate.segments,
                        rules,
                    )
                    all_reports.extend(reports)
                    raw.append(
                        (
                            output,
                            candidate.log_score,
                            CandidateDerivation(
                                derivation_id=f"{parent_node_id}:{concept_id}:{candidate.candidate_id}",
                                child_candidate_ids=(candidate.candidate_id,),
                                rule_ids=tuple(rule.rule_id for rule in rules),
                                note="single-child propagation due to missing sibling concept",
                            ),
                        )
                    )
            output_distributions.append(
                normalize_and_prune(
                    parent_node_id,
                    concept_id,
                    raw,
                    beam_width=self.beam_width,
                )
            )
        output_beam = NodeBeamState(
            node_id=parent_node_id,
            distributions=tuple(output_distributions),
            beam_width=self.beam_width,
            source_child_ids=(left.node_id, right.node_id),
        )
        return ReconstructionStep(
            parent_node_id=parent_node_id,
            child_node_ids=(left.node_id, right.node_id),
            input_beams=(left, right),
            output_beam=output_beam,
            rule_reports=tuple(all_reports),
            anomaly_reports=tuple(anomalies),
        )
