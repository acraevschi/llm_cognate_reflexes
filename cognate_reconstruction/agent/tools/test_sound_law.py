"""Adapter for strict parsing and exact deterministic rule diffs."""

from __future__ import annotations

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.schemas import TestSoundLawArgs, TestSoundLawResult
from cognate_reconstruction.rules.parser import parse_rule
from cognate_reconstruction.schemas.common import WorkbenchModel


def test_sound_law(
    raw_arguments: WorkbenchModel,
    context: AgentContext,
    call_id: str,
) -> TestSoundLawResult:
    arguments = TestSoundLawArgs.model_validate(raw_arguments)
    unknown = sorted(set(arguments.source_child_ids) - set(context.child_ids))
    if unknown:
        raise ValueError(f"rule targets inactive children: {unknown}")
    rule = parse_rule(arguments.dsl)
    selected_concepts = set(arguments.concept_ids)
    forms = tuple(
        form
        for child_id in arguments.source_child_ids
        for form in context.lexicon(
            child_id, arguments.segmentation_overlay_id
        ).forms
        if not selected_concepts or form.concept_id in selected_concepts
    )
    if not forms:
        raise ValueError("no forms matched the requested child and concept scope")
    anchors_by_concept: dict[str, dict[str, tuple[str, ...]]] = {}
    for anchor in context.anchors:
        anchors_by_concept.setdefault(anchor.concept_id, {})[anchor.form_id] = anchor.segments
    anchor_expected = {
        form.form_id: anchors_by_concept.get(form.concept_id, {}) for form in forms
    }
    report = context.rule_engine.apply_rule(
        rule,
        forms,
        anchor_expected=anchor_expected,
    )
    supporting = tuple(result.form_id for result in report.results if result.locations)
    result = TestSoundLawResult(
        validation_call_id=call_id,
        parsed_rule=rule,
        source_child_ids=arguments.source_child_ids,
        segmentation_overlay_id=arguments.segmentation_overlay_id,
        report=report,
        supporting_form_ids=supporting,
    )
    context.validations[call_id] = result
    return result
