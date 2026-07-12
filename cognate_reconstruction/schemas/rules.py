"""Sound-law AST, application diff, and anomaly schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, computed_field, model_validator

from cognate_reconstruction.schemas.common import (
    MORPHOLOGICAL_BOUNDARY_TOKENS,
    NonEmptyStr,
    WorkbenchModel,
)


class SegmentExpression(WorkbenchModel):
    """Literal token expression with explicit structural-boundary vocabulary."""

    tokens: tuple[NonEmptyStr, ...]
    morphological_boundary_tokens: frozenset[str] = MORPHOLOGICAL_BOUNDARY_TOKENS

    @model_validator(mode="after")
    def validate_boundary_vocabulary(self) -> SegmentExpression:
        if self.morphological_boundary_tokens - MORPHOLOGICAL_BOUNDARY_TOKENS:
            raise ValueError("only '+' and '-' are supported as morphological boundaries")
        return self


class RuleEnvironment(WorkbenchModel):
    left: SegmentExpression | None = None
    right: SegmentExpression | None = None
    word_initial: bool = False
    word_final: bool = False


class ParsedSoundRule(WorkbenchModel):
    rule_id: NonEmptyStr
    source: NonEmptyStr
    target: SegmentExpression
    replacement: SegmentExpression
    environment: RuleEnvironment

    @model_validator(mode="after")
    def validate_target(self) -> ParsedSoundRule:
        if not self.target.tokens:
            raise ValueError("sound-rule target must not be empty")
        if any(t in self.target.morphological_boundary_tokens for t in self.target.tokens):
            raise ValueError("morphological boundaries may constrain context but not be targets")
        if any(t in self.replacement.morphological_boundary_tokens for t in self.replacement.tokens):
            raise ValueError("rules may not insert morphological boundaries")
        return self


class ReconstructionRule(WorkbenchModel):
    """A confidence-weighted rule scoped to one or more active children."""

    rule: ParsedSoundRule
    source_child_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_child_scope(self) -> ReconstructionRule:
        if len(set(self.source_child_ids)) != len(self.source_child_ids):
            raise ValueError("source_child_ids must be unique")
        return self


class AnchorPolicy(StrEnum):
    """How optional ancestor anchors influence reconstruction."""

    IGNORE = "ignore"
    ADVISORY = "advisory"
    SCORED = "scored"


class ApplicationStatus(StrEnum):
    APPLIED = "applied"
    TARGET_ABSENT = "target_absent"
    CONTEXT_MISMATCH = "context_mismatch"
    ANCHOR_MISMATCH = "anchor_mismatch"


class MatchLocation(WorkbenchModel):
    start_token: int = Field(ge=0)
    end_token: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_span(self) -> MatchLocation:
        if self.end_token <= self.start_token:
            raise ValueError("match location must be a non-empty half-open span")
        return self


class FormRuleResult(WorkbenchModel):
    form_id: NonEmptyStr
    source_candidate_id: NonEmptyStr | None = None
    input_segments: tuple[str, ...]
    output_segments: tuple[str, ...]
    status: ApplicationStatus
    locations: tuple[MatchLocation, ...] = ()
    target_occurrences: int = Field(default=0, ge=0)
    anchor_ids: tuple[NonEmptyStr, ...] = ()
    matched_anchor_ids: tuple[NonEmptyStr, ...] = ()
    explanation: NonEmptyStr


class RuleApplicationReport(WorkbenchModel):
    rule: ParsedSoundRule
    results: tuple[FormRuleResult, ...]

    @computed_field
    @property
    def words_applied(self) -> int:
        return sum(bool(result.locations) for result in self.results)

    @computed_field
    @property
    def anchors_matched(self) -> int:
        return sum(len(result.matched_anchor_ids) for result in self.results)

    @computed_field
    @property
    def exceptions(self) -> tuple[FormRuleResult, ...]:
        return tuple(
            result for result in self.results if result.status is not ApplicationStatus.APPLIED
        )


class AnomalyType(StrEnum):
    LOANWORD = "loanword"
    MORPHOLOGICAL_LEVELING = "morphological_leveling"
    TABOO_DEFORMATION = "taboo_deformation"
    UNKNOWN_IRREGULARITY = "unknown_irregularity"


class AnomalyReport(WorkbenchModel):
    anomaly_type: AnomalyType
    explanation: NonEmptyStr
    form_id: NonEmptyStr | None = None
    concept_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def require_subject(self) -> AnomalyReport:
        if self.form_id is None and self.concept_id is None:
            raise ValueError("an anomaly must identify a form_id or concept_id")
        return self
