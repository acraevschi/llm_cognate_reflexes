"""Probability distributions retained by the deterministic beam search."""

from __future__ import annotations

import math

from pydantic import Field, model_validator

from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel


class CandidateDerivation(WorkbenchModel):
    derivation_id: NonEmptyStr
    child_candidate_ids: tuple[NonEmptyStr, ...]
    rule_ids: tuple[NonEmptyStr, ...] = ()
    alignment_ids: tuple[NonEmptyStr, ...] = ()
    note: NonEmptyStr | None = None


class ReconstructionCandidate(WorkbenchModel):
    candidate_id: NonEmptyStr
    segments: tuple[NonEmptyStr, ...] = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    log_score: float
    derivations: tuple[CandidateDerivation, ...]

    @model_validator(mode="after")
    def validate_finite_scores(self) -> ReconstructionCandidate:
        if not math.isfinite(self.probability) or not math.isfinite(self.log_score):
            raise ValueError("candidate probability and log_score must be finite")
        return self


class ConceptCandidateDistribution(WorkbenchModel):
    concept_id: NonEmptyStr
    candidates: tuple[ReconstructionCandidate, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_distribution(self) -> ConceptCandidateDistribution:
        candidate_ids = [c.candidate_id for c in self.candidates]
        segment_sequences = [c.segments for c in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique within a distribution")
        if len(set(segment_sequences)) != len(segment_sequences):
            raise ValueError("candidate segment sequences must be unique within a distribution")
        if abs(sum(c.probability for c in self.candidates) - 1.0) > 1e-8:
            raise ValueError("candidate probabilities must sum to one")
        if any(
            left.probability < right.probability
            for left, right in zip(self.candidates, self.candidates[1:])
        ):
            raise ValueError("candidates must be sorted by descending probability")
        return self


class NodeBeamState(WorkbenchModel):
    node_id: NonEmptyStr
    distributions: tuple[ConceptCandidateDistribution, ...]
    beam_width: int = Field(ge=1)
    source_child_ids: tuple[NonEmptyStr, ...] | None = None

    @model_validator(mode="after")
    def validate_beam(self) -> NodeBeamState:
        concepts = [distribution.concept_id for distribution in self.distributions]
        if len(concepts) != len(set(concepts)):
            raise ValueError("a node beam may contain each concept only once")
        if any(len(d.candidates) > self.beam_width for d in self.distributions):
            raise ValueError("distribution exceeds beam_width")
        if self.source_child_ids is not None and len(self.source_child_ids) < 2:
            raise ValueError("a reconstructed node must identify at least two children")
        return self
