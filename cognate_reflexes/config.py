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
            per dataset. A finite cap enables bounded-memory streaming.
        max_total_binary_trees: Maximum fully resolved Newick trees retained
            per family while sampling polytomy resolutions.
        historical_lineages_path: CSV manifest of curated historical
            ancestor-to-descendant branch relations.
        include_historical: Whether to emit reconstruction examples for
            validated historical targets.
        min_historical_age_gap: Required age difference between a historical
            target and a dated descendant, measured in source units before
            present.  ``0`` requires only strict temporal ordering.
        temporal_trees_path: Optional CSV mapping datasets to authoritative
            time-aware Newick files.  Standard CLDF TreeTables and
            ``cldf/tree.nwk`` are discovered automatically as well.
        include_temporal_trees: Whether to derive historical targets directly
            from authoritative time-aware Newick trees.
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
    max_triplets_per_dataset: int | None = 5000  # Cap on generated triplets
    max_total_binary_trees: int = 64
    historical_lineages_path: str = "./data/historical_lineages.csv"
    include_historical: bool = True
    min_historical_age_gap: float = 0.0
    temporal_trees_path: str = "./data/temporal_trees.csv"
    include_temporal_trees: bool = True
    seed: int | None = 42
