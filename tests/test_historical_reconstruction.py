from __future__ import annotations

from pathlib import Path

from cognate_reflexes.config import Config
from cognate_reflexes.data.historical import HistoricalLineageManifest
from cognate_reflexes.data.loader import DatasetForms
from cognate_reflexes.data.temporal_trees import (
    TemporalTreeManifest,
    discover_temporal_lineages,
)
from cognate_reflexes.examples.historical_reconstruction import (
    generate_historical_reconstruction_examples,
)
from cognate_reflexes.examples.models import Form, LanguageData


def _language(
    variety_id: str,
    name: str,
    *,
    historical: bool = False,
    date: float | None = None,
) -> LanguageData:
    return LanguageData(
        glottocode=variety_id,
        name=name,
        variety_id=variety_id,
        tree_glottocode=variety_id,
        is_historical=historical,
        date_before_present=date,
        family="Test",
    )


def _dataset() -> DatasetForms:
    target_id = "toy:old"
    left_id = "toy:left"
    right_id = "toy:right"
    cognatesets = [f"toy:{index}" for index in range(12)]
    languages = {
        target_id: _language(target_id, "Old Toy", historical=True, date=1000),
        left_id: _language(left_id, "Left Toy", date=500),
        right_id: _language(right_id, "Right Toy", date=400),
    }
    forms_by_language = {}
    for variety_id in languages:
        forms_by_language[variety_id] = {
            cognateset_id: [
                Form(
                    form_id=f"{variety_id}:{cognateset_id}",
                    language=variety_id,
                    language_name=languages[variety_id].name,
                    segments=("t", "a"),
                    concept=cognateset_id,
                    concepticon_id=None,
                    cognateset_id=cognateset_id,
                    dataset="toy",
                    tree_glottocode=variety_id,
                )
            ]
            for cognateset_id in cognatesets
        }
    return DatasetForms(
        dataset_name="toy",
        forms_by_language=forms_by_language,
        languages=languages,
        cognate_coverage={
            cognateset_id: set(languages) for cognateset_id in cognatesets
        },
        proto_languages=set(),
        historical_languages={target_id},
        family="Test",
    )


def _manifest(path: Path, rows: list[str]) -> HistoricalLineageManifest:
    path.write_text(
        "dataset,target_variety_id,branch_id,descendant_variety_id,evidence\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
    return HistoricalLineageManifest.from_csv(path)


def test_historical_examples_require_two_distinct_branches(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path / "lineages.csv",
        [
            "toy,toy:old,left,toy:left,unit test",
            "toy,toy:old,right,toy:right,unit test",
        ],
    )
    examples = list(
        generate_historical_reconstruction_examples(
            _dataset(),
            manifest,
            Config(task="reconstruction", min_cognates=10, max_cognates=12),
        )
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.metadata.target_kind == "historical"
    assert set(example.metadata.historical_branch_ids) == {"left", "right"}
    assert example.metadata.variety_ids == ("toy:left", "toy:right", "toy:old")
    assert example.masked_indices == list(range(12))


def test_single_descendant_branch_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path / "lineages.csv",
        [
            "toy,toy:old,polish_chain,toy:left,unit test",
            "toy,toy:old,polish_chain,toy:right,unit test",
        ],
    )
    examples = list(
        generate_historical_reconstruction_examples(
            _dataset(),
            manifest,
            Config(task="reconstruction", min_cognates=10, max_cognates=12),
        )
    )

    assert examples == []


def test_manifest_can_nominate_historical_target_without_source_flag(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    dataset.languages["toy:old"].is_historical = False
    dataset.historical_languages.clear()
    manifest = _manifest(
        tmp_path / "lineages.csv",
        [
            "toy,toy:old,left,toy:left,unit test",
            "toy,toy:old,right,toy:right,unit test",
        ],
    )

    examples = list(
        generate_historical_reconstruction_examples(
            dataset,
            manifest,
            Config(task="reconstruction", min_cognates=10, max_cognates=12),
        )
    )

    assert len(examples) == 1
    assert examples[0].target.is_historical


def test_internal_old_node_in_source_newick_generates_example(
    tmp_path: Path,
) -> None:
    dataset = _dataset()
    dataset.languages["toy:old"].is_historical = False
    dataset.historical_languages.clear()
    dataset.source_path = tmp_path
    cldf_dir = tmp_path / "cldf"
    cldf_dir.mkdir()
    (cldf_dir / "tree.nwk").write_text(
        "((left,right)old,unrelated)root;",
        encoding="utf-8",
    )

    lineages = discover_temporal_lineages(dataset, TemporalTreeManifest())
    assert set(lineages) == {"toy:old"}
    assert len(lineages["toy:old"]) == 2

    examples = list(
        generate_historical_reconstruction_examples(
            dataset,
            HistoricalLineageManifest(),
            Config(task="reconstruction", min_cognates=10, max_cognates=12),
            automatic_lineages=lineages,
        )
    )

    assert len(examples) == 1
    assert examples[0].target.is_historical
    assert all(
        branch_id.startswith("source_tree:")
        for branch_id in examples[0].metadata.historical_branch_ids
    )
