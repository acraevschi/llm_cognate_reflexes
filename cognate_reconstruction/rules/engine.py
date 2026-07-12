"""Deterministic token-level sound-rule application with explicit diffs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import (
    ApplicationStatus,
    FormRuleResult,
    MatchLocation,
    ParsedSoundRule,
    RuleApplicationReport,
)


def _occurrences(sequence: tuple[str, ...], target: tuple[str, ...]) -> tuple[int, ...]:
    width = len(target)
    return tuple(
        index
        for index in range(len(sequence) - width + 1)
        if sequence[index : index + width] == target
    )


def _context_matches(rule: ParsedSoundRule, sequence: tuple[str, ...], start: int) -> bool:
    environment = rule.environment
    end = start + len(rule.target.tokens)
    if environment.word_initial and start != 0:
        return False
    if environment.word_final and end != len(sequence):
        return False
    if environment.left is not None:
        left = environment.left.tokens
        if start < len(left) or sequence[start - len(left) : start] != left:
            return False
    if environment.right is not None:
        right = environment.right.tokens
        if sequence[end : end + len(right)] != right:
            return False
    return True


def _replace(
    sequence: tuple[str, ...],
    target: tuple[str, ...],
    replacement: tuple[str, ...],
    starts: tuple[int, ...],
) -> tuple[str, ...]:
    selected: list[int] = []
    next_available = 0
    for start in starts:
        if start >= next_available:
            selected.append(start)
            next_available = start + len(target)
    output: list[str] = []
    cursor = 0
    for start in selected:
        output.extend(sequence[cursor:start])
        output.extend(replacement)
        cursor = start + len(target)
    output.extend(sequence[cursor:])
    return tuple(output)


AnchorExpectation: TypeAlias = Mapping[
    str,
    tuple[str, ...] | Mapping[str, tuple[str, ...]],
]


def _anchors_for_form(
    anchors: AnchorExpectation,
    form_id: str,
) -> Mapping[str, tuple[str, ...]]:
    expected = anchors.get(form_id)
    if expected is None:
        return {}
    if isinstance(expected, tuple):
        # Backward-compatible form: the transformed form doubles as anchor ID.
        return {form_id: expected}
    return expected


class RuleEngine:
    """Apply parsed rules without any dependency on tree traversal."""

    def apply_rule(
        self,
        rule: ParsedSoundRule,
        forms: Sequence[LexicalForm],
        *,
        anchor_expected: AnchorExpectation | None = None,
        source_candidate_ids: Mapping[str, str] | None = None,
    ) -> RuleApplicationReport:
        anchors = anchor_expected or {}
        candidate_ids = source_candidate_ids or {}
        results: list[FormRuleResult] = []
        for form in forms:
            occurrences = _occurrences(form.segments, rule.target.tokens)
            matching = tuple(
                start for start in occurrences if _context_matches(rule, form.segments, start)
            )
            output = _replace(form.segments, rule.target.tokens, rule.replacement.tokens, matching)
            form_anchors = _anchors_for_form(anchors, form.form_id)
            anchor_ids = tuple(sorted(form_anchors))
            # An unchanged form is not an anchor match caused by this rule.
            matched_anchor_ids = tuple(
                anchor_id
                for anchor_id, expected in sorted(form_anchors.items())
                if matching and output == expected
            )
            if not occurrences:
                status = ApplicationStatus.TARGET_ABSENT
                explanation = "target sequence is absent"
            elif not matching:
                status = ApplicationStatus.CONTEXT_MISMATCH
                explanation = "target occurs, but never in the specified environment"
            elif anchor_ids and not matched_anchor_ids:
                status = ApplicationStatus.ANCHOR_MISMATCH
                explanation = "rule applied mechanically, but did not produce the anchor form"
            else:
                status = ApplicationStatus.APPLIED
                explanation = f"rule applied at {len(matching)} location(s)"
            results.append(
                FormRuleResult(
                    form_id=form.form_id,
                    source_candidate_id=candidate_ids.get(form.form_id),
                    input_segments=form.segments,
                    output_segments=output,
                    status=status,
                    locations=tuple(
                        MatchLocation(
                            start_token=start,
                            end_token=start + len(rule.target.tokens),
                        )
                        for start in matching
                    ),
                    target_occurrences=len(occurrences),
                    anchor_ids=anchor_ids,
                    matched_anchor_ids=matched_anchor_ids,
                    explanation=explanation,
                )
            )
        return RuleApplicationReport(rule=rule, results=tuple(results))

    def apply_rules(
        self,
        rules: Sequence[ParsedSoundRule],
        forms: Sequence[LexicalForm],
        *,
        anchor_expected: AnchorExpectation | None = None,
        source_candidate_ids: Mapping[str, str] | None = None,
    ) -> tuple[tuple[LexicalForm, ...], tuple[RuleApplicationReport, ...]]:
        """Apply an ordered rule cascade, returning final forms and every diff."""
        current = tuple(forms)
        reports: list[RuleApplicationReport] = []
        for rule in rules:
            report = self.apply_rule(
                rule,
                current,
                anchor_expected=anchor_expected,
                source_candidate_ids=source_candidate_ids,
            )
            reports.append(report)
            current = tuple(
                LexicalForm(
                    form_id=form.form_id,
                    variety_id=form.variety_id,
                    concept_id=form.concept_id,
                    segments=result.output_segments,
                    cognate_set_id=form.cognate_set_id,
                    morphological_boundary_tokens=form.morphological_boundary_tokens,
                    provenance=form.provenance,
                )
                for form, result in zip(current, report.results, strict=True)
            )
        return current, tuple(reports)
