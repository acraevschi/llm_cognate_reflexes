"""Normalized lexical input schemas."""

from __future__ import annotations

from pydantic import Field, model_validator

from cognate_reconstruction.schemas.common import (
    MORPHOLOGICAL_BOUNDARY_TOKENS,
    NonEmptyStr,
    WorkbenchModel,
)


class ConceptMetadata(WorkbenchModel):
    """Optional human-readable semantics for a stable concept identifier."""

    concept_id: NonEmptyStr
    gloss: NonEmptyStr | None = None
    concepticon_id: NonEmptyStr | None = None
    aliases: tuple[NonEmptyStr, ...] = ()
    semantic_field: NonEmptyStr | None = None

    @model_validator(mode="after")
    def validate_aliases(self) -> ConceptMetadata:
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("concept aliases must be unique")
        return self


class FormProvenance(WorkbenchModel):
    dataset_id: NonEmptyStr | None = None
    source_form_id: NonEmptyStr | None = None
    source_row: int | None = Field(default=None, ge=1)
    segment_source: NonEmptyStr | None = None


class LexicalForm(WorkbenchModel):
    """A tokenized form; ``+`` and ``-`` are structural, not IPA segments."""

    form_id: NonEmptyStr
    variety_id: NonEmptyStr
    concept_id: NonEmptyStr
    segments: tuple[NonEmptyStr, ...] = Field(min_length=1)
    cognate_set_id: NonEmptyStr | None = None
    morphological_boundary_tokens: frozenset[str] = MORPHOLOGICAL_BOUNDARY_TOKENS
    provenance: FormProvenance = Field(default_factory=FormProvenance)

    @model_validator(mode="after")
    def validate_boundaries(self) -> LexicalForm:
        if not self.morphological_boundary_tokens:
            raise ValueError("morphological_boundary_tokens must not be empty")
        if self.morphological_boundary_tokens - MORPHOLOGICAL_BOUNDARY_TOKENS:
            raise ValueError("only '+' and '-' are supported as morphological boundaries")
        return self

    @property
    def phonetic_segments(self) -> tuple[str, ...]:
        """Return segments with structural morphological boundaries removed."""
        return tuple(s for s in self.segments if s not in self.morphological_boundary_tokens)


class LanguageLexicon(WorkbenchModel):
    variety_id: NonEmptyStr
    name: NonEmptyStr
    forms: tuple[LexicalForm, ...]
    tree_glottocode: NonEmptyStr | None = None
    family: NonEmptyStr | None = None
    is_historical: bool = False

    @model_validator(mode="after")
    def validate_form_ownership(self) -> LanguageLexicon:
        ids: set[str] = set()
        for form in self.forms:
            if form.variety_id != self.variety_id:
                raise ValueError(f"form {form.form_id!r} belongs to another variety")
            if form.form_id in ids:
                raise ValueError(f"duplicate form_id {form.form_id!r}")
            ids.add(form.form_id)
        return self
