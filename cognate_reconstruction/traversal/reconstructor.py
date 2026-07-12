"""Deterministic rule-driven combination of an n-ary child beam set."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from cognate_reconstruction.rules.engine import RuleEngine
from cognate_reconstruction.schemas.beam import CandidateDerivation, NodeBeamState
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import (
    AnchorPolicy,
    AnomalyReport,
    ParsedSoundRule,
    ReconstructionRule,
    RuleApplicationReport,
)
from cognate_reconstruction.schemas.traversal import (
    NodeReconstructionContext,
    ReconstructionStep,
)
from cognate_reconstruction.traversal.beam import RawCandidate, normalize_and_prune


@dataclass(frozen=True)
class _TransformedCandidate:
    segments: tuple[str, ...]
    confidence_log_score: float
    applied_rule_ids: tuple[str, ...]
    matched_anchor_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PartialCombination:
    """A bounded Cartesian-product state over children processed so far."""

    outputs: tuple[tuple[str, ...], ...]
    anchor_matches: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]
    log_score: float
    derivations: tuple[CandidateDerivation, ...]


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _extend_anchor_matches(
    existing: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
    output: tuple[str, ...],
    anchor_ids: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    matches = {segments: set(ids) for segments, ids in existing}
    matches.setdefault(output, set()).update(anchor_ids)
    return tuple(
        (segments, tuple(sorted(ids)))
        for segments, ids in sorted(matches.items())
    )


def _merge_and_prune_partials(
    partials: Sequence[_PartialCombination],
    *,
    beam_width: int,
    anchor_match_log_boost: float,
) -> list[_PartialCombination]:
    """Merge equivalent partial states and prune after each child expansion."""
    grouped: dict[
        tuple[
            tuple[tuple[str, ...], ...],
            tuple[tuple[tuple[str, ...], tuple[str, ...]], ...],
        ],
        list[_PartialCombination],
    ] = defaultdict(list)
    for partial in partials:
        grouped[(partial.outputs, partial.anchor_matches)].append(partial)
    merged = [
        _PartialCombination(
            outputs=outputs,
            anchor_matches=anchor_matches,
            log_score=_logsumexp([item.log_score for item in items]),
            # Bound provenance growth along with hypothesis growth.
            derivations=tuple(
                derivation
                for item in items
                for derivation in item.derivations
            )[:beam_width],
        )
        for (outputs, anchor_matches), items in grouped.items()
    ]
    return sorted(
        merged,
        key=lambda item: (
            -(
                item.log_score
                + max((len(ids) for _, ids in item.anchor_matches), default=0)
                * anchor_match_log_boost
            ),
            item.outputs,
            item.anchor_matches,
        ),
    )[:beam_width]


class RuleBasedReconstructor:
    """Apply branch-scoped rule cascades and combine n-ary child evidence.

    Rules are interpreted in their written direction. If historical inference
    requires inverse rules, callers must supply those inverse hypotheses
    explicitly; the engine never guesses an inverse for a non-bijective law.

    Anchors are advisory by default: matches remain visible in reports without
    changing scores. With ``anchor_policy=SCORED``, ``anchor_match_factor`` is a
    likelihood multiplier applied once per unique match before pruning.
    """

    def __init__(
        self,
        *,
        beam_width: int = 5,
        anchor_policy: AnchorPolicy | str = AnchorPolicy.ADVISORY,
        anchor_match_factor: float = 100.0,
        engine: RuleEngine | None = None,
    ) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be positive")
        if not math.isfinite(anchor_match_factor) or anchor_match_factor < 1.0:
            raise ValueError("anchor_match_factor must be finite and at least 1")
        self.beam_width = beam_width
        self.anchor_policy = AnchorPolicy(anchor_policy)
        self.anchor_match_factor = anchor_match_factor
        self.anchor_match_log_boost = (
            math.log(anchor_match_factor)
            if self.anchor_policy is AnchorPolicy.SCORED
            else 0.0
        )
        self.engine = engine or RuleEngine()

    @staticmethod
    def _scope_rules(
        rules: Sequence[ReconstructionRule | ParsedSoundRule],
        child_ids: tuple[str, ...],
    ) -> tuple[ReconstructionRule, ...]:
        active = set(child_ids)
        scoped: list[ReconstructionRule] = []
        for rule in rules:
            normalized = (
                rule
                if isinstance(rule, ReconstructionRule)
                else ReconstructionRule(
                    rule=rule,
                    source_child_ids=child_ids,
                    confidence=1.0,
                )
            )
            unknown = sorted(set(normalized.source_child_ids) - active)
            if unknown:
                raise ValueError(
                    f"rule {normalized.rule.rule_id!r} targets inactive children: {unknown}"
                )
            scoped.append(normalized)
        return tuple(scoped)

    def _transform(
        self,
        parent_node_id: str,
        concept_id: str,
        child_id: str,
        candidate_id: str,
        segments: tuple[str, ...],
        rules: Sequence[ReconstructionRule],
        anchors: Sequence[LexicalForm],
    ) -> tuple[_TransformedCandidate, tuple[RuleApplicationReport, ...]]:
        form = LexicalForm(
            form_id=f"beam-form:{candidate_id}",
            variety_id=parent_node_id,
            concept_id=concept_id,
            segments=segments,
        )
        child_rules = tuple(rule for rule in rules if child_id in rule.source_child_ids)
        active_anchors = () if self.anchor_policy is AnchorPolicy.IGNORE else anchors
        anchor_expected = {
            form.form_id: {anchor.form_id: anchor.segments for anchor in active_anchors}
        }
        transformed, reports = self.engine.apply_rules(
            tuple(rule.rule for rule in child_rules),
            (form,),
            anchor_expected=anchor_expected,
            source_candidate_ids={form.form_id: candidate_id},
        )
        applied_rule_ids: list[str] = []
        confidence_score = 0.0
        matched_anchor_ids: set[str] = set()
        for scoped_rule, report in zip(child_rules, reports, strict=True):
            result = report.results[0]
            if result.locations:
                applied_rule_ids.append(scoped_rule.rule.rule_id)
                confidence_score += math.log(scoped_rule.confidence)
            matched_anchor_ids.update(result.matched_anchor_ids)
        final_segments = transformed[0].segments
        final_anchor_ids = {
            anchor.form_id
            for anchor in active_anchors
            if anchor.segments == final_segments
        }
        matched_anchor_ids.intersection_update(final_anchor_ids)
        return (
            _TransformedCandidate(
                segments=final_segments,
                confidence_log_score=confidence_score,
                applied_rule_ids=tuple(applied_rule_ids),
                matched_anchor_ids=tuple(sorted(matched_anchor_ids)),
            ),
            reports,
        )

    def reconstruct(
        self,
        parent_node_id: str,
        children: Sequence[NodeBeamState],
        *,
        rules: Sequence[ReconstructionRule | ParsedSoundRule] = (),
        anomalies: Sequence[AnomalyReport] = (),
        anchors: Sequence[LexicalForm] = (),
        evidence_context: NodeReconstructionContext | None = None,  # noqa: ARG002
    ) -> ReconstructionStep:
        child_beams = tuple(children)
        if len(child_beams) < 2:
            raise ValueError("reconstruction requires at least two child beams")
        child_ids = tuple(child.node_id for child in child_beams)
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("child beam node IDs must be unique")
        scoped_rules = self._scope_rules(rules, child_ids)

        distributions_by_child = tuple(
            {distribution.concept_id: distribution for distribution in child.distributions}
            for child in child_beams
        )
        concept_ids = sorted(
            set().union(*(set(distributions) for distributions in distributions_by_child))
        )
        anchors_by_concept: dict[str, list[LexicalForm]] = defaultdict(list)
        for anchor in anchors:
            anchors_by_concept[anchor.concept_id].append(anchor)

        output_distributions = []
        all_reports: list[RuleApplicationReport] = []
        for concept_id in concept_ids:
            available = [
                (child, distributions[concept_id])
                for child, distributions in zip(
                    child_beams, distributions_by_child, strict=True
                )
                if concept_id in distributions
            ]
            partials: list[_PartialCombination] = []
            for child_index, (child, distribution) in enumerate(available):
                transformed_candidates = []
                for candidate in distribution.candidates:
                    transformed, reports = self._transform(
                        parent_node_id,
                        concept_id,
                        child.node_id,
                        candidate.candidate_id,
                        candidate.segments,
                        scoped_rules,
                        anchors_by_concept[concept_id],
                    )
                    all_reports.extend(reports)
                    transformed_candidates.append((candidate, transformed))

                expanded: list[_PartialCombination] = []
                if child_index == 0:
                    for candidate, transformed in transformed_candidates:
                        note = "single child evidence"
                        if transformed.matched_anchor_ids:
                            note += "; matched anchors: " + ", ".join(
                                transformed.matched_anchor_ids
                            )
                        expanded.append(
                            _PartialCombination(
                                outputs=(transformed.segments,),
                                anchor_matches=(
                                    (
                                        transformed.segments,
                                        transformed.matched_anchor_ids,
                                    ),
                                ),
                                log_score=candidate.log_score
                                + transformed.confidence_log_score,
                                derivations=(
                                    CandidateDerivation(
                                        derivation_id=(
                                            f"{parent_node_id}:{concept_id}:"
                                            f"{candidate.candidate_id}"
                                        ),
                                        child_candidate_ids=(candidate.candidate_id,),
                                        rule_ids=transformed.applied_rule_ids,
                                        note=note,
                                    ),
                                ),
                            )
                        )
                else:
                    for partial in partials:
                        for candidate, transformed in transformed_candidates:
                            outputs = tuple(
                                sorted(set(partial.outputs) | {transformed.segments})
                            )
                            anchor_matches = _extend_anchor_matches(
                                partial.anchor_matches,
                                transformed.segments,
                                transformed.matched_anchor_ids,
                            )
                            derivations = tuple(
                                CandidateDerivation(
                                    derivation_id=(
                                        f"{derivation.derivation_id}:"
                                        f"{candidate.candidate_id}"
                                    ),
                                    child_candidate_ids=(
                                        *derivation.child_candidate_ids,
                                        candidate.candidate_id,
                                    ),
                                    rule_ids=_ordered_unique(
                                        (
                                            *derivation.rule_ids,
                                            *transformed.applied_rule_ids,
                                        )
                                    ),
                                    note="incremental n-ary child-beam derivation",
                                )
                                for derivation in partial.derivations
                            )
                            expanded.append(
                                _PartialCombination(
                                    outputs=outputs,
                                    anchor_matches=anchor_matches,
                                    log_score=(
                                        partial.log_score
                                        + candidate.log_score
                                        + transformed.confidence_log_score
                                    ),
                                    derivations=derivations,
                                )
                            )
                partials = _merge_and_prune_partials(
                    expanded,
                    beam_width=self.beam_width,
                    anchor_match_log_boost=self.anchor_match_log_boost,
                )

            raw: list[RawCandidate] = []
            for partial in partials:
                branch_penalty = -math.log(len(partial.outputs))
                anchor_matches = dict(partial.anchor_matches)
                for output in partial.outputs:
                    raw.append(
                        (
                            output,
                            partial.log_score
                            + branch_penalty
                            + len(anchor_matches.get(output, ()))
                            * self.anchor_match_log_boost,
                            partial.derivations[0],
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
            source_child_ids=child_ids,
        )
        return ReconstructionStep(
            parent_node_id=parent_node_id,
            child_node_ids=child_ids,
            input_beams=child_beams,
            output_beam=output_beam,
            rule_reports=tuple(all_reports),
            anomaly_reports=tuple(anomalies),
        )
