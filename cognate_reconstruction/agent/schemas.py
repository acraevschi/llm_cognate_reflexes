"""Strict provider-neutral schemas for the hypothesis-manager layer."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from cognate_reconstruction.schemas.alignment import MultipleAlignmentMap
from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.lexicon import ConceptMetadata
from cognate_reconstruction.schemas.rules import (
    AnomalyReport,
    AnchorPolicy,
    ParsedSoundRule,
    ReconstructionRule,
    RuleApplicationReport,
)
from cognate_reconstruction.schemas.traversal import (
    EvidenceKind,
    EvidenceRelation,
)


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class LLMToolCall(WorkbenchModel):
    call_id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, Any]


class LLMMessage(WorkbenchModel):
    role: MessageRole
    content: str | None = None
    tool_calls: tuple[LLMToolCall, ...] = ()
    tool_call_id: NonEmptyStr | None = None
    name: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> LLMMessage:
        if self.role is MessageRole.ASSISTANT:
            if self.content is None and not self.tool_calls:
                raise ValueError("assistant messages need content or tool calls")
        elif self.tool_calls:
            raise ValueError("only assistant messages may contain tool calls")
        if self.role is MessageRole.TOOL:
            if self.tool_call_id is None or self.name is None or self.content is None:
                raise ValueError("tool messages require call ID, name, and content")
        elif self.tool_call_id is not None or self.name is not None:
            raise ValueError("tool_call_id and name are only valid on tool messages")
        if self.role in {MessageRole.SYSTEM, MessageRole.USER} and self.content is None:
            raise ValueError("system and user messages require content")
        return self


class LLMToolDefinition(WorkbenchModel):
    name: NonEmptyStr
    description: NonEmptyStr
    parameters: dict[str, Any]


class ToolError(WorkbenchModel):
    error_type: NonEmptyStr
    message: NonEmptyStr


class ToolExecutionResult(WorkbenchModel):
    ok: bool
    result: dict[str, Any] | None = None
    error: ToolError | None = None

    @model_validator(mode="after")
    def validate_result_shape(self) -> ToolExecutionResult:
        if self.ok == (self.error is not None):
            raise ValueError("successful results cannot contain errors and failures must")
        if self.ok != (self.result is not None):
            raise ValueError("successful tool calls must contain a result")
        return self


class GetAlignmentsArgs(WorkbenchModel):
    node_ids: tuple[NonEmptyStr, ...] = Field(min_length=2)
    concept_ids: tuple[NonEmptyStr, ...] = ()
    form_ids: tuple[NonEmptyStr, ...] = ()
    segmentation_overlay_id: NonEmptyStr | None = None
    respect_cognate_sets: bool = True
    include_anchors: bool = False

    @model_validator(mode="after")
    def validate_nodes(self) -> GetAlignmentsArgs:
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("get_alignments requires distinct node IDs")
        return self


class GetAlignmentsResult(WorkbenchModel):
    alignment_map: MultipleAlignmentMap
    segmentation_overlay_id: NonEmptyStr | None = None


class EvidenceScope(StrEnum):
    ACTIVE_CHILDREN = "active_children"
    AVAILABLE_TREE = "available_tree"


class SegmentPosition(StrEnum):
    INITIAL = "initial"
    FINAL = "final"
    CONTAINS = "contains"
    EXACT = "exact"


class ListConceptsArgs(WorkbenchModel):
    query: str | None = None
    scope: EvidenceScope = EvidenceScope.ACTIVE_CHILDREN
    node_ids: tuple[NonEmptyStr, ...] = ()
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)


class ConceptListing(WorkbenchModel):
    concept: ConceptMetadata
    form_count: int = Field(ge=1)
    node_ids: tuple[NonEmptyStr, ...]


class ListConceptsResult(WorkbenchModel):
    concepts: tuple[ConceptListing, ...]
    next_offset: int | None = Field(default=None, ge=0)


class SearchFormsArgs(WorkbenchModel):
    scope: EvidenceScope = EvidenceScope.ACTIVE_CHILDREN
    node_ids: tuple[NonEmptyStr, ...] = ()
    concept_ids: tuple[NonEmptyStr, ...] = ()
    concept_query: str | None = None
    segment_pattern: tuple[NonEmptyStr, ...] = ()
    position: SegmentPosition = SegmentPosition.CONTAINS
    cognate_set_ids: tuple[NonEmptyStr, ...] = ()
    include_boundaries: bool = False
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=200)


class FormSearchHit(WorkbenchModel):
    node_id: NonEmptyStr
    evidence_kind: EvidenceKind
    relation: EvidenceRelation
    concept: ConceptMetadata
    form: LexicalForm


class SearchFormsResult(WorkbenchModel):
    hits: tuple[FormSearchHit, ...]
    next_offset: int | None = Field(default=None, ge=0)


class ListAvailableNodesArgs(WorkbenchModel):
    kinds: tuple[EvidenceKind, ...] = ()
    relations: tuple[EvidenceRelation, ...] = ()


class AvailableNodeSummary(WorkbenchModel):
    node_id: NonEmptyStr
    kind: EvidenceKind
    relation: EvidenceRelation
    descendant_leaf_ids: tuple[NonEmptyStr, ...] = ()
    form_count: int = Field(ge=0)
    concept_count: int = Field(ge=0)


class ListAvailableNodesResult(WorkbenchModel):
    nodes: tuple[AvailableNodeSummary, ...]


class TestSoundLawArgs(WorkbenchModel):
    dsl: NonEmptyStr
    source_child_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    concept_ids: tuple[NonEmptyStr, ...] = ()
    segmentation_overlay_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_scope(self) -> TestSoundLawArgs:
        if len(set(self.source_child_ids)) != len(self.source_child_ids):
            raise ValueError("source_child_ids must be unique")
        return self


class TestSoundLawResult(WorkbenchModel):
    validation_call_id: NonEmptyStr
    parsed_rule: ParsedSoundRule
    source_child_ids: tuple[NonEmptyStr, ...]
    segmentation_overlay_id: NonEmptyStr | None = None
    report: RuleApplicationReport
    supporting_form_ids: tuple[NonEmptyStr, ...]


class MorphemeSegmentation(WorkbenchModel):
    form_id: NonEmptyStr
    segments: tuple[NonEmptyStr, ...] = Field(min_length=1)


class SegmentMorphemesArgs(WorkbenchModel):
    segmentations: tuple[MorphemeSegmentation, ...] = Field(min_length=1)
    rationale: NonEmptyStr
    base_overlay_id: NonEmptyStr | None = None


class SegmentMorphemesResult(WorkbenchModel):
    segmentation_overlay_id: NonEmptyStr
    forms: tuple[LexicalForm, ...]
    rationale: NonEmptyStr


class CommittedSoundRule(WorkbenchModel):
    rule_id: NonEmptyStr
    dsl: NonEmptyStr
    direction: Literal["child_to_parent"] = "child_to_parent"
    source_child_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    confidence: float = Field(gt=0.0, le=1.0)
    validation_call_id: NonEmptyStr
    supporting_form_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    rationale: NonEmptyStr

    @model_validator(mode="after")
    def validate_references(self) -> CommittedSoundRule:
        if len(set(self.source_child_ids)) != len(self.source_child_ids):
            raise ValueError("source_child_ids must be unique")
        if len(set(self.supporting_form_ids)) != len(self.supporting_form_ids):
            raise ValueError("supporting_form_ids must be unique")
        return self


class CommitReconstructionArgs(WorkbenchModel):
    node_id: NonEmptyStr
    segmentation_overlay_id: NonEmptyStr | None = None
    rules: tuple[CommittedSoundRule, ...]
    anomalies: tuple[AnomalyReport, ...]
    summary: NonEmptyStr

    @model_validator(mode="after")
    def validate_unique_rule_ids(self) -> CommitReconstructionArgs:
        ids = [rule.rule_id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("committed rule IDs must be unique")
        return self


class CommittedReconstruction(WorkbenchModel):
    request: CommitReconstructionArgs
    parsed_rules: tuple[ReconstructionRule, ...]


class CommitReconstructionResult(WorkbenchModel):
    status: Literal["committed"] = "committed"
    reconstruction: CommittedReconstruction


class NodeLexiconSummary(WorkbenchModel):
    node_id: NonEmptyStr
    name: NonEmptyStr
    form_count: int = Field(ge=0)
    concept_count: int = Field(ge=0)


class NodePromptPayload(WorkbenchModel):
    node_id: NonEmptyStr
    active_children: tuple[NodeLexiconSummary, ...] = Field(min_length=2)
    anchor_policy: AnchorPolicy = AnchorPolicy.ADVISORY
    anchors: tuple[LexicalForm, ...] = ()
