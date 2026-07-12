"""Create immutable, session-local morphological segmentation overlays."""

from __future__ import annotations

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.schemas import (
    SegmentMorphemesArgs,
    SegmentMorphemesResult,
)
from cognate_reconstruction.schemas.common import MORPHOLOGICAL_BOUNDARY_TOKENS, WorkbenchModel


def segment_morphemes(
    raw_arguments: WorkbenchModel,
    context: AgentContext,
    call_id: str,  # noqa: ARG001 - uniform tool signature
) -> SegmentMorphemesResult:
    arguments = SegmentMorphemesArgs.model_validate(raw_arguments)
    ids = [item.form_id for item in arguments.segmentations]
    if len(ids) != len(set(ids)):
        raise ValueError("a segmentation request may edit each form only once")
    base_forms = context.forms_for_overlay(arguments.base_overlay_id)
    edited = []
    for segmentation in arguments.segmentations:
        try:
            original = base_forms[segmentation.form_id]
        except KeyError as error:
            raise ValueError(f"unknown form {segmentation.form_id!r}") from error
        phonetic = tuple(
            segment
            for segment in segmentation.segments
            if segment not in MORPHOLOGICAL_BOUNDARY_TOKENS
        )
        if phonetic != original.phonetic_segments:
            raise ValueError(
                f"segmentation for {segmentation.form_id!r} changes phonetic tokens"
            )
        edited.append(original.model_copy(update={"segments": segmentation.segments}))
    forms = tuple(edited)
    overlay_id = context.store_overlay(
        forms,
        base_overlay_id=arguments.base_overlay_id,
    )
    return SegmentMorphemesResult(
        segmentation_overlay_id=overlay_id,
        forms=forms,
        rationale=arguments.rationale,
    )
