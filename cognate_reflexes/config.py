"""Central configuration for the cognate_reflexes pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration for the cognate_reflexes pipeline.

    Attributes:
        data_dir: Root directory containing Lexibank CLDF datasets.
        glottolog_dir: Path to a local Glottolog clone.
        task: Pipeline task — ``"cognate_reflex"`` or ``"reconstruction"``.
        min_cognates: Minimum number of shared cognate sets required for a
            triplet to be emitted.
        max_cognates: Maximum number of cognate sets to include per triplet.
        mask_ratio: Fraction of target forms to mask in the cognate-reflex
            task (typical range 0.20–0.50).
        max_branch_length: Maximum total branch length from MRCA to either
            input language.  ``None`` means no limit.
        max_edge_distance: Fallback edge-count limit when branch lengths are
            unavailable.  ``None`` means no limit.
        include_glosses: Whether to include Concepticon glosses in the
            formatted output.
        show_language_names: Whether to reveal language names to the model.
            Disabled by default so the model cannot memorise identity cues.
        max_polytomy_resolutions: Upper bound on binary resolutions
            enumerated per polytomy node.
        max_triplets_per_dataset: Maximum number of triplets to generate
            per dataset (useful to prevent explosion in large families).
        seed: Random seed for reproducibility.  ``None`` disables seeding.
    """

    data_dir: str = "./data/lexibank"
    glottolog_dir: str = "./data/glottolog"
    task: str = "cognate_reflex"  # or "reconstruction"
    min_cognates: int = 15
    max_cognates: int = 100
    mask_ratio: float = 0.3  # For cognate reflex task (20-50% range)
    max_branch_length: float | None = None  # Distance limit for reflex triplets
    max_edge_distance: int | None = None  # Fallback when no branch lengths
    include_glosses: bool = True
    show_language_names: bool = False  # No language identity by default
    max_polytomy_resolutions: int = 50  # Cap on binary resolutions per polytomy
    max_triplets_per_dataset: int | None = None  # Cap on generated triplets
    seed: int | None = 42
