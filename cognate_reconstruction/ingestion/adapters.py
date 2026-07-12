"""Compatibility adapters for the existing cognate_reflexes loader models."""

from __future__ import annotations

from cognate_reconstruction.schemas.lexicon import (
    ConceptMetadata,
    FormProvenance,
    LanguageLexicon,
    LexicalForm,
)
from cognate_reflexes.data.loader import DatasetForms


def adapt_concept_metadata(dataset: DatasetForms) -> tuple[ConceptMetadata, ...]:
    """Recover readable glosses for deterministic concept search."""
    glosses: dict[str, set[str]] = {}
    concepticon_ids: dict[str, str] = {}
    for forms_by_cognate in dataset.forms_by_language.values():
        for forms in forms_by_cognate.values():
            for form in forms:
                concept_id = form.concepticon_id or form.concept
                glosses.setdefault(concept_id, set()).add(form.concept)
                if form.concepticon_id is not None:
                    concepticon_ids[concept_id] = form.concepticon_id
    concepts = []
    for concept_id, names in sorted(glosses.items()):
        ordered = sorted(names)
        concepts.append(
            ConceptMetadata(
                concept_id=concept_id,
                gloss=ordered[0],
                concepticon_id=concepticon_ids.get(concept_id),
                aliases=tuple(ordered[1:]),
            )
        )
    return tuple(concepts)


def adapt_dataset_forms(dataset: DatasetForms) -> tuple[LanguageLexicon, ...]:
    """Convert a legacy ``DatasetForms`` without changing identity semantics."""
    lexicons: list[LanguageLexicon] = []
    for variety_id in sorted(dataset.languages):
        language = dataset.languages[variety_id]
        forms: list[LexicalForm] = []
        for cognate_set_id in sorted(dataset.forms_by_language.get(variety_id, {})):
            for form in dataset.forms_by_language[variety_id][cognate_set_id]:
                concept_id = form.concepticon_id or form.concept
                forms.append(
                    LexicalForm(
                        form_id=form.form_id,
                        variety_id=variety_id,
                        concept_id=concept_id,
                        segments=form.segments,
                        cognate_set_id=form.cognateset_id,
                        provenance=FormProvenance(
                            dataset_id=form.dataset,
                            source_form_id=form.form_id,
                            segment_source=form.segment_source,
                        ),
                    )
                )
        if forms:
            lexicons.append(
                LanguageLexicon(
                    variety_id=variety_id,
                    name=language.name,
                    forms=tuple(forms),
                    tree_glottocode=language.tree_glottocode,
                    family=language.family,
                    is_historical=language.is_historical,
                )
            )
    return tuple(lexicons)
