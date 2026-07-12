"""LingPy-independent alignment and correspondence schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel


class AlignmentMember(WorkbenchModel):
    form_id: NonEmptyStr
    variety_id: NonEmptyStr
    concept_id: NonEmptyStr
    cognate_set_id: NonEmptyStr | None = None
    aligned_segments: tuple[str | None, ...]
    is_anchor: bool = False


class AlignmentResult(WorkbenchModel):
    alignment_id: NonEmptyStr
    concept_id: NonEmptyStr
    members: tuple[AlignmentMember, ...] = Field(min_length=2)
    alignment_score: float | None = None
    method: Literal["sca", "lexstat"] = "sca"
    mode: Literal["global", "local", "overlap", "dialign"] = "global"

    @model_validator(mode="after")
    def validate_width(self) -> AlignmentResult:
        widths = {len(member.aligned_segments) for member in self.members}
        if len(widths) != 1:
            raise ValueError("all alignment members must have equal width")
        return self


class CorrespondenceObservation(WorkbenchModel):
    alignment_id: NonEmptyStr
    column_index: int = Field(ge=0)
    left_segment: str | None
    right_segment: str | None
    left_context: tuple[str | None, str | None]
    right_context: tuple[str | None, str | None]
    anchor_supported: bool = False


class CorrespondenceSummary(WorkbenchModel):
    left_segment: str | None
    right_segment: str | None
    count: int = Field(ge=1)
    anchor_count: int = Field(ge=0)
    observations: tuple[CorrespondenceObservation, ...]

    @model_validator(mode="after")
    def validate_counts(self) -> CorrespondenceSummary:
        if self.count != len(self.observations) or self.anchor_count > self.count:
            raise ValueError("correspondence counts disagree with observations")
        return self


class CorrespondenceMap(WorkbenchModel):
    left_variety_id: NonEmptyStr
    right_variety_id: NonEmptyStr
    alignments: tuple[AlignmentResult, ...]
    correspondences: tuple[CorrespondenceSummary, ...]
