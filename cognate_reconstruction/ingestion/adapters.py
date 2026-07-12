"""Compatibility adapters for the existing cognate_reflexes loader models."""

from __future__ import annotations

from cognate_reconstruction.schemas.lexicon import (
    FormProvenance,
    LanguageLexicon,
    LexicalForm,
)
from cognate_reflexes.data.loader import DatasetForms


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
