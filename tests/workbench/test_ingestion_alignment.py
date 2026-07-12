from __future__ import annotations

import pytest

from cognate_reconstruction.alignment import LingPyAligner
from cognate_reconstruction.ingestion import ingest_payload
from cognate_reconstruction.schemas.ingestion import TreeOrigin, WorkbenchPayload
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm


def lexicon(variety_id: str, segments: tuple[str, ...]) -> LanguageLexicon:
    return LanguageLexicon(
        variety_id=variety_id,
        name=variety_id,
        forms=(
            LexicalForm(
                form_id=f"{variety_id}-water",
                variety_id=variety_id,
                concept_id="water",
                segments=segments,
                cognate_set_id="water-1",
            ),
        ),
    )


def test_provided_tree_leaf_validation() -> None:
    payload = WorkbenchPayload(
        lexicons=(lexicon("A", ("p", "a")), lexicon("B", ("b", "a"))),
        newick="(A,B);",
    )
    ingested = ingest_payload(payload)
    assert ingested.tree.origin is TreeOrigin.PROVIDED
    with pytest.raises(ValueError, match="leaf mismatch"):
        ingest_payload(payload.model_copy(update={"newick": "(A,C);"}))


def test_lingpy_alignment_returns_correspondences() -> None:
    result = LingPyAligner().align(
        lexicon("A", ("p", "a")), lexicon("B", ("b", "a"))
    )
    assert len(result.alignments) == 1
    pairs = {(item.left_segment, item.right_segment) for item in result.correspondences}
    assert ("p", "b") in pairs
    assert ("a", "a") in pairs


def test_lingpy_tree_induction() -> None:
    payload = WorkbenchPayload(
        lexicons=(
            lexicon("A", ("p", "a")),
            lexicon("B", ("b", "a")),
            lexicon("C", ("k", "i")),
        ),
    )
    ingested = ingest_payload(payload)
    assert ingested.tree.origin is TreeOrigin.INDUCED
    assert set(ingested.tree.leaf_variety_ids) == {"A", "B", "C"}
    assert ingested.tree.newick.endswith(";")
