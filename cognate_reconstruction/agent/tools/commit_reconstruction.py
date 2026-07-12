"""Validated terminal tool for committing a node reconstruction."""

from __future__ import annotations

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.schemas import (
    CommitReconstructionArgs,
    CommitReconstructionResult,
    CommittedReconstruction,
)
from cognate_reconstruction.rules.parser import parse_rule
from cognate_reconstruction.schemas.common import WorkbenchModel
from cognate_reconstruction.schemas.rules import ReconstructionRule


def commit_reconstruction(
    raw_arguments: WorkbenchModel,
    context: AgentContext,
    call_id: str,  # noqa: ARG001 - uniform tool signature
) -> CommitReconstructionResult:
    arguments = CommitReconstructionArgs.model_validate(raw_arguments)
    if context.commit is not None:
        raise ValueError("this node already has a committed reconstruction")
    if arguments.node_id != context.node_id:
        raise ValueError(
            f"commit node {arguments.node_id!r} does not match active node {context.node_id!r}"
        )
    if (
        arguments.segmentation_overlay_id is not None
        and arguments.segmentation_overlay_id not in context.overlays
    ):
        raise ValueError(
            f"unknown segmentation overlay {arguments.segmentation_overlay_id!r}"
        )
    active_children = set(context.child_ids)
    parsed_rules: list[ReconstructionRule] = []
    for committed in arguments.rules:
        unknown = sorted(set(committed.source_child_ids) - active_children)
        if unknown:
            raise ValueError(f"rule {committed.rule_id!r} targets inactive children: {unknown}")
        try:
            validation = context.validations[committed.validation_call_id]
        except KeyError as error:
            raise ValueError(
                f"rule {committed.rule_id!r} references an unknown validation call"
            ) from error
        parsed = parse_rule(committed.dsl, rule_id=committed.rule_id)
        if parsed.source != validation.parsed_rule.source:
            raise ValueError(
                f"rule {committed.rule_id!r} was not validated with this exact DSL"
            )
        if set(committed.source_child_ids) != set(validation.source_child_ids):
            raise ValueError(
                f"rule {committed.rule_id!r} was not validated for this child scope"
            )
        if validation.segmentation_overlay_id != arguments.segmentation_overlay_id:
            raise ValueError(
                f"rule {committed.rule_id!r} was not validated on the committed "
                "segmentation overlay"
            )
        unsupported = sorted(
            set(committed.supporting_form_ids) - set(validation.supporting_form_ids)
        )
        if unsupported:
            raise ValueError(
                f"rule {committed.rule_id!r} cites unsupported forms: {unsupported}"
            )
        parsed_rules.append(
            ReconstructionRule(
                rule=parsed,
                source_child_ids=committed.source_child_ids,
                confidence=committed.confidence,
            )
        )

    active_form_ids = {form.form_id for form in context.all_forms} | {
        anchor.form_id for anchor in context.anchors
    }
    active_concept_ids = {form.concept_id for form in context.all_forms} | {
        anchor.concept_id for anchor in context.anchors
    }
    for anomaly in arguments.anomalies:
        if anomaly.form_id is not None and anomaly.form_id not in active_form_ids:
            raise ValueError(f"anomaly references unknown form {anomaly.form_id!r}")
        if anomaly.concept_id is not None and anomaly.concept_id not in active_concept_ids:
            raise ValueError(f"anomaly references unknown concept {anomaly.concept_id!r}")

    reconstruction = CommittedReconstruction(
        request=arguments,
        parsed_rules=tuple(parsed_rules),
    )
    context.commit = reconstruction
    return CommitReconstructionResult(reconstruction=reconstruction)
