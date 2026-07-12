"""Initial payload orchestration."""

from __future__ import annotations

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
    expected = {lexicon.variety_id for lexicon in payload.lexicons}
    if payload.newick is None:
        tree = induce_tree(payload.lexicons, method=payload.tree_method)
    else:
        root = parse_newick(payload.newick)
        actual = root.get_leaf_labels()
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"tree/lexicon leaf mismatch: missing={missing}, extra={extra}")
        tree = TreeArtifact(
            newick=payload.newick.strip(),
            origin=TreeOrigin.PROVIDED,
            leaf_variety_ids=tuple(sorted(actual)),
        )
    return IngestedDataset(lexicons=payload.lexicons, tree=tree)
