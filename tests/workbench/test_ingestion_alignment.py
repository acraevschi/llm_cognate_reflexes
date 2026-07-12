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


def test_lingpy_alignment_supports_three_way_msa() -> None:
    result = LingPyAligner().align_multiple(
        (
            lexicon("A", ("p", "a")),
            lexicon("B", ("b", "a")),
            lexicon("C", ("f", "a")),
        )
    )
    assert result.variety_ids == ("A", "B", "C")
    assert len(result.alignments[0].members) == 3
    assert len(result.pairwise_correspondences) == 3


def test_lingpy_alignment_respects_known_cognate_sets_by_default() -> None:
    left = lexicon("A", ("p", "a"))
    right = lexicon("B", ("b", "a"))
    left = left.model_copy(
        update={
            "forms": (
                left.forms[0].model_copy(update={"cognate_set_id": "cog-1"}),
            )
        }
    )
    right = right.model_copy(
        update={
            "forms": (
                right.forms[0].model_copy(update={"cognate_set_id": "cog-2"}),
            )
        }
    )
    assert not LingPyAligner().align_multiple((left, right)).alignments
    assert len(
        LingPyAligner().align_multiple(
            (left, right), respect_cognate_sets=False
        ).alignments
    ) == 1


def test_lingpy_alignment_rejects_unimplemented_lexstat_label() -> None:
    with pytest.raises(ValueError, match="only the SCA"):
        LingPyAligner(method="lexstat")  # type: ignore[arg-type]


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


def test_lingpy_tree_induction_quotes_dataset_scoped_ids() -> None:
    payload = WorkbenchPayload(
        lexicons=(
            lexicon("dataset:A", ("p", "a")),
            lexicon("dataset:B", ("b", "a")),
            lexicon("dataset:C", ("k", "i")),
        ),
    )
    ingested = ingest_payload(payload)
    assert ingested.tree.leaf_variety_ids == (
        "dataset:A",
        "dataset:B",
        "dataset:C",
    )
    assert "'dataset:A'" in ingested.tree.newick
