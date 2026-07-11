"""Core data models for the cognate_reflexes triplet pipeline.

Every class here is a plain :func:`dataclasses.dataclass` — no ORM, no
validation library — so the models stay lightweight and easy to serialise.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Form:
    """A single lexical form tied to a concept and a cognate set.

    Attributes:
        form_id: Unique form identifier within the source dataset.
        language: Glottocode of the language this form belongs to.
        language_name: Human-readable language name (metadata only).
        segments: IPA segments as a tuple (e.g. ``("b", "a", "n"...)``).
        concept: Concept gloss (e.g. ``"WATER"``), from Concepticon or
            the dataset's own parameter name.
        concepticon_id: Concepticon numeric identifier, or ``None``
            if unmapped.
        cognateset_id: Dataset-scoped cognate-set identifier
            (prefixed with dataset name).
        dataset: Name of the Lexibank dataset this form came from.
    """

    form_id: str
    language: str
    language_name: str
    segments: tuple[str, ...]
    concept: str
    concepticon_id: str | None
    cognateset_id: str
    dataset: str


@dataclass
class LanguageData:
    """Linguistic data for a single language (doculect) within one dataset.

    Attributes:
        glottocode: Glottolog identifier for the language.
        name: Human-readable language name (metadata only — not shown to
            the model by default).
        forms: Lexical forms associated with this language for a given
            triplet.
        latitude: Geographic latitude, if known.
        longitude: Geographic longitude, if known.
        family: Top-level language-family name, if known.
        is_proto: Whether this language is identified as a
            proto-language (reconstructed ancestor).
    """

    glottocode: str
    name: str
    forms: list[Form] = field(default_factory=list)
    latitude: float | None = None
    longitude: float | None = None
    family: str | None = None
    subgroup: str | None = None
    is_proto: bool = False


@dataclass
class ExampleMetadata:
    """Provenance and structural metadata attached to every example.

    Attributes:
        source_dataset: The single Lexibank dataset the example was
            extracted from (no cross-dataset merging).
        language_family: Top-level language family.
        tree_depth: Depth of the Most Recent Common Ancestor (MRCA) node
            in the reference tree.
        branch_lengths: Summed branch length from MRCA to each input.
        num_cognate_sets: Number of cognate sets shared across all
            languages in this example.
        glottocodes: ``(*inputs, target)`` Glottocodes.
        coordinates: Mapping from glottocode to ``(lat, lon)`` or ``None``.
        concept_ids: Concepticon IDs used in this example.
        cognateset_ids: Cognate-set IDs used in this example.
    """

    source_dataset: str
    language_family: str
    tree_depth: int
    branch_lengths: list[float | None]
    num_cognate_sets: int
    glottocodes: tuple[str, ...]
    coordinates: dict[str, tuple[float, float] | None]
    concept_ids: list[str]
    cognateset_ids: list[str]


@dataclass
class TrainingExample:
    """A single training example consisting of N input languages and aligned
    cognate forms for 1 target language.

    For the *cognate_reflex* task some target forms are masked and the
    model must predict them.  For *reconstruction* the target is an
    ancestral proto-language and all its forms are hidden.

    Attributes:
        task: ``"cognate_reflex"`` or ``"reconstruction"``.
        inputs: List of input language data.
        target: Target language data (forms to predict).
        masked_indices: Indices into ``target.forms`` that are masked
            (the model must predict these).
        metadata: Structural / provenance metadata for this example.
    """

    task: str
    inputs: list[LanguageData]
    target: LanguageData
    masked_indices: list[int]
    metadata: ExampleMetadata

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def revealed_forms(self) -> list[Form]:
        """Target forms that are visible to the model (not masked)."""
        return [
            f for i, f in enumerate(self.target.forms) if i not in self.masked_indices
        ]

    @property
    def masked_forms(self) -> list[Form]:
        """Target forms that are hidden from the model (to be predicted)."""
        return [
            f for i, f in enumerate(self.target.forms) if i in self.masked_indices
        ]
