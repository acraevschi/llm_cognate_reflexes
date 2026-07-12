"""LexStat-backed language-distance calculation and tree induction."""

from __future__ import annotations

from typing import Literal

from cognate_reconstruction.schemas.ingestion import DistanceMatrix, TreeArtifact, TreeOrigin
from cognate_reconstruction.schemas.lexicon import LanguageLexicon


def _lexstat_rows(lexicons: tuple[LanguageLexicon, ...]) -> dict[int, list[object]]:
    rows: dict[int, list[object]] = {
        0: ["doculect", "concept", "ipa", "tokens", "cogid"]
    }
    index = 1
    cognate_ids: dict[tuple[str, object], int] = {}
    for lexicon in lexicons:
        for form in lexicon.forms:
            if form.cognate_set_id is not None:
                cognate_key: tuple[str, object] = ("explicit", form.cognate_set_id)
            else:
                cognate_key = ("fallback", (form.concept_id, form.phonetic_segments))
            cogid = cognate_ids.setdefault(cognate_key, len(cognate_ids) + 1)
            rows[index] = [
                lexicon.variety_id,
                form.concept_id,
                "".join(form.phonetic_segments),
                list(form.phonetic_segments),
                cogid,
            ]
            index += 1
    return rows


def induce_tree(
    lexicons: tuple[LanguageLexicon, ...],
    *,
    method: Literal["neighbor", "upgma"] = "neighbor",
) -> TreeArtifact:
    """Infer a Newick tree using LexStat cognate distances and NJ/UPGMA."""
    if len(lexicons) < 2:
        raise ValueError("tree induction requires at least two lexicons")
    if any(not lexicon.forms for lexicon in lexicons):
        raise ValueError("tree induction requires at least one form per lexicon")

    from lingpy import LexStat  # type: ignore[import-untyped]
    from lingpy.algorithm.clustering import matrix2tree  # type: ignore[import-untyped]

    lexstat = LexStat(_lexstat_rows(lexicons))
    raw_matrix = lexstat.get_distances(method="sca", aggregate=True)
    taxa = tuple(str(taxon) for taxon in lexstat.cols)
    values = tuple(tuple(float(value) for value in row) for row in raw_matrix)
    distance_matrix = DistanceMatrix(taxa=taxa, values=values, method="lexstat-sca")
    tree = matrix2tree([list(row) for row in values], list(taxa), tree_calc=method)
    newick = str(tree).strip()
    if not newick.endswith(";"):
        newick += ";"
    return TreeArtifact(
        newick=newick,
        origin=TreeOrigin.INDUCED,
        leaf_variety_ids=taxa,
        induction_method=method,
        distance_matrix=distance_matrix,
    )
