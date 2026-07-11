"""Curated lineage relations for historical reconstruction targets.

Glottolog classification is not a temporal phylogeny: a historical language
is commonly represented as a leaf.  This module therefore keeps the
historical ancestor-to-descendant evidence separate from tree lookup.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HistoricalLineage:
    """One descendant variety assigned to a first-diverging child branch."""

    dataset: str
    target_variety_id: str
    branch_id: str
    descendant_variety_id: str
    evidence: str = ""


class HistoricalLineageManifest:
    """Read and index curated historical lineage relations from CSV.

    The manifest intentionally names unique dataset-scoped variety IDs.  This
    keeps historical stages distinct even when a source reuses a Glottocode
    for several stages of one lineage.
    """

    REQUIRED_COLUMNS = {
        "dataset",
        "target_variety_id",
        "branch_id",
        "descendant_variety_id",
    }

    def __init__(self, relations: list[HistoricalLineage] | None = None) -> None:
        self._relations = relations or []
        self._by_dataset_target: dict[
            tuple[str, str], dict[str, set[str]]
        ] = defaultdict(lambda: defaultdict(set))
        for relation in self._relations:
            self._by_dataset_target[
                (relation.dataset, relation.target_variety_id)
            ][relation.branch_id].add(relation.descendant_variety_id)

    @classmethod
    def from_csv(cls, path: str | Path) -> "HistoricalLineageManifest":
        """Load a CSV manifest, returning an empty one when it is absent."""
        csv_path = Path(path)
        if not csv_path.exists():
            return cls()

        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or [])
            missing = cls.REQUIRED_COLUMNS - headers
            if missing:
                joined = ", ".join(sorted(missing))
                raise ValueError(
                    f"Historical lineage manifest {csv_path} is missing: {joined}"
                )

            relations = []
            for row in reader:
                values = {
                    column: (row.get(column) or "").strip()
                    for column in cls.REQUIRED_COLUMNS
                }
                if not all(values.values()):
                    continue
                relations.append(
                    HistoricalLineage(
                        dataset=values["dataset"],
                        target_variety_id=values["target_variety_id"],
                        branch_id=values["branch_id"],
                        descendant_variety_id=values["descendant_variety_id"],
                        evidence=(row.get("evidence") or "").strip(),
                    )
                )
        return cls(relations)

    def branches_for(
        self, dataset: str, target_variety_id: str
    ) -> dict[str, set[str]]:
        """Return a defensive copy of the target's branch membership."""
        branches = self._by_dataset_target.get((dataset, target_variety_id), {})
        return {branch: set(descendants) for branch, descendants in branches.items()}

    def targets_for(self, dataset: str) -> list[str]:
        """Return sorted historical target IDs registered for one dataset."""
        return sorted(
            target
            for source_dataset, target in self._by_dataset_target
            if source_dataset == dataset
        )

    def datasets(self) -> set[str]:
        """Return datasets with at least one explicitly curated target."""
        return {dataset for dataset, _ in self._by_dataset_target}
