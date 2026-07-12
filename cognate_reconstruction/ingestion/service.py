"""Initial payload orchestration."""

from __future__ import annotations

from cognate_reconstruction.ingestion.tree_normalization import normalize_tree, to_newick
from cognate_reconstruction.ingestion.tree_induction import induce_tree
from cognate_reconstruction.schemas.ingestion import (
    IngestedDataset,
    TreeArtifact,
    TreeOrigin,
    WorkbenchPayload,
)
from cognate_reflexes.tree.newick_utils import parse_newick


def ingest_payload(payload: WorkbenchPayload) -> IngestedDataset:
    """Validate a supplied tree or induce one when it is absent."""
    usable_lexicons = tuple(lexicon for lexicon in payload.lexicons if lexicon.forms)
    expected = {lexicon.variety_id for lexicon in usable_lexicons}
    if len(expected) < 2:
        raise ValueError("ingestion requires at least two lexicons with usable forms")
    if payload.newick is None:
        tree = induce_tree(usable_lexicons, method=payload.tree_method)
    else:
        root = parse_newick(payload.newick)
        actual = root.get_leaf_labels()
        if missing := sorted(expected - actual):
            raise ValueError(f"tree/lexicon leaf mismatch: missing={missing}")
        normalized = normalize_tree(root, expected)
        tree = TreeArtifact(
            newick=to_newick(normalized),
            origin=TreeOrigin.PROVIDED,
            leaf_variety_ids=tuple(sorted(expected)),
        )
    return IngestedDataset(
        lexicons=usable_lexicons,
        tree=tree,
        concepts=payload.concepts,
    )
