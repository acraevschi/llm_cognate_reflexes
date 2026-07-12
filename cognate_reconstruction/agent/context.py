"""Mutable, node-local state used only by deterministic tool adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from cognate_reconstruction.alignment.protocol import AlignmentProvider
from cognate_reconstruction.rules.engine import RuleEngine
from cognate_reconstruction.schemas.rules import AnchorPolicy
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm
from cognate_reconstruction.schemas.lexicon import ConceptMetadata
from cognate_reconstruction.schemas.traversal import NodeEvidence

from .schemas import CommittedReconstruction, TestSoundLawResult


@dataclass
class AgentContext:
    node_id: str
    child_lexicons: tuple[LanguageLexicon, ...]
    aligner: AlignmentProvider
    anchors: tuple[LexicalForm, ...] = ()
    anchor_policy: AnchorPolicy = AnchorPolicy.ADVISORY
    evidence: tuple[NodeEvidence, ...] = ()
    concepts: tuple[ConceptMetadata, ...] = ()
    rule_engine: RuleEngine = field(default_factory=RuleEngine)
    overlays: dict[str, dict[str, LexicalForm]] = field(default_factory=dict)
    validations: dict[str, TestSoundLawResult] = field(default_factory=dict)
    commit: CommittedReconstruction | None = None

    def __post_init__(self) -> None:
        ids = [lexicon.variety_id for lexicon in self.child_lexicons]
        if len(ids) < 2 or len(ids) != len(set(ids)):
            raise ValueError("an agent context needs at least two distinct children")

    @property
    def child_ids(self) -> tuple[str, ...]:
        return tuple(lexicon.variety_id for lexicon in self.child_lexicons)

    @property
    def all_forms(self) -> tuple[LexicalForm, ...]:
        return tuple(form for lexicon in self.child_lexicons for form in lexicon.forms)

    def evidence_lexicon(
        self,
        node_id: str,
        overlay_id: str | None = None,
    ) -> LanguageLexicon:
        if node_id in self.child_ids:
            return self.lexicon(node_id, overlay_id)
        try:
            return next(item.lexicon for item in self.evidence if item.node_id == node_id)
        except StopIteration as error:
            raise ValueError(f"unknown or unavailable evidence node {node_id!r}") from error

    def forms_for_overlay(self, overlay_id: str | None) -> dict[str, LexicalForm]:
        forms = {form.form_id: form for form in self.all_forms}
        if overlay_id is None:
            return forms
        try:
            forms.update(self.overlays[overlay_id])
        except KeyError as error:
            raise ValueError(f"unknown segmentation overlay {overlay_id!r}") from error
        return forms

    def lexicon(self, child_id: str, overlay_id: str | None = None) -> LanguageLexicon:
        try:
            original = next(
                lexicon for lexicon in self.child_lexicons if lexicon.variety_id == child_id
            )
        except StopIteration as error:
            raise ValueError(f"unknown child {child_id!r}") from error
        forms = self.forms_for_overlay(overlay_id)
        return original.model_copy(
            update={"forms": tuple(forms[form.form_id] for form in original.forms)}
        )

    def store_overlay(
        self,
        forms: tuple[LexicalForm, ...],
        *,
        base_overlay_id: str | None,
    ) -> str:
        base = dict(self.overlays.get(base_overlay_id, {})) if base_overlay_id else {}
        base.update({form.form_id: form for form in forms})
        material = "\n".join(
            f"{form_id}\t{' '.join(form.segments)}"
            for form_id, form in sorted(base.items())
        )
        overlay_id = f"seg-{hashlib.sha256(material.encode()).hexdigest()[:12]}"
        self.overlays[overlay_id] = base
        return overlay_id
