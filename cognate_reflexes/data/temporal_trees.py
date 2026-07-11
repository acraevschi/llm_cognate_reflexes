"""Discovery of authoritative time-aware Newick trees in source datasets."""

from __future__ import annotations

import base64
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote_to_bytes

from pycldf import Dataset

from cognate_reflexes.data.loader import DatasetForms
from cognate_reflexes.tree.newick_utils import TreeNode, parse_newick

logger = logging.getLogger(__name__)

_HISTORICAL_NAME = re.compile(
    r"\b(?:old|middle|ancient|classical|early|late|pre)[ -]",
    re.IGNORECASE,
)


class TemporalTreeManifest:
    """Optional mapping of datasets to authoritative Newick files.

    A source tree path is relative to the Lexibank dataset root unless it is
    absolute.  The CSV intentionally makes provenance explicit for trees that
    are not published through CLDF's TreeTable/MediaTable mechanism.
    """

    def __init__(self, tree_paths: dict[str, list[str]] | None = None) -> None:
        self._tree_paths = tree_paths or {}

    @classmethod
    def from_csv(cls, path: str | Path) -> "TemporalTreeManifest":
        csv_path = Path(path)
        if not csv_path.exists():
            return cls()
        tree_paths: dict[str, list[str]] = defaultdict(list)
        with csv_path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"dataset", "tree_path"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Temporal tree manifest {csv_path} is missing: "
                    f"{', '.join(sorted(missing))}"
                )
            for row in reader:
                dataset = (row.get("dataset") or "").strip()
                tree_path = (row.get("tree_path") or "").strip()
                if dataset and tree_path:
                    tree_paths[dataset].append(tree_path)
        return cls(dict(tree_paths))

    def paths_for(self, dataset: str) -> list[str]:
        return list(self._tree_paths.get(dataset, []))

    def datasets(self) -> set[str]:
        return set(self._tree_paths)


def _walk(node: TreeNode) -> Iterator[TreeNode]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _label_keys(label: str) -> set[str]:
    """Return conservative matching keys for common Newick label encodings."""
    normalized = " ".join(label.strip().replace("_", " ").split()).casefold()
    compact = re.sub(r"[^\w]", "", normalized)
    return {key for key in {label.strip(), normalized, compact} if key}


class _LanguageLabelResolver:
    """Resolve source-tree labels to one unambiguous dataset variety ID."""

    def __init__(self, dataset: DatasetForms) -> None:
        self._labels: dict[str, set[str]] = defaultdict(set)
        for variety_id, language in dataset.languages.items():
            local_id = variety_id.split(":", 1)[-1]
            for label in (
                variety_id,
                local_id,
                language.glottocode,
                language.tree_glottocode or "",
                language.name,
            ):
                for key in _label_keys(label):
                    self._labels[key].add(variety_id)

    def resolve(self, label: str | None) -> str | None:
        if not label:
            return None
        candidates: set[str] = set()
        for key in _label_keys(label):
            candidates.update(self._labels.get(key, set()))
        return next(iter(candidates)) if len(candidates) == 1 else None


def _data_url_to_text(url: str) -> str | None:
    """Decode the Newick payload used by CLDF MediaTable data URLs."""
    if not url.startswith("data:") or "," not in url:
        return None
    header, payload = url.split(",", 1)
    try:
        raw = (
            base64.b64decode(payload)
            if ";base64" in header.lower()
            else unquote_to_bytes(payload)
        )
        return raw.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def _metadata_path(dataset_path: Path) -> Path | None:
    cldf_dir = dataset_path / "cldf"
    if cldf_dir.is_dir():
        candidates = sorted(cldf_dir.glob("*-metadata.json"))
        if candidates:
            return candidates[0]
        simple = cldf_dir / "cldf-metadata.json"
        if simple.exists():
            return simple
    return None


def _cldf_tree_strings(dataset_path: Path) -> list[str]:
    """Read Newick trees published through CLDF TreeTable and MediaTable."""
    metadata_path = _metadata_path(dataset_path)
    if metadata_path is None:
        return []
    try:
        cldf = Dataset.from_metadata(metadata_path)
        media_by_id = {row.get("ID"): row for row in cldf["MediaTable"]}
        trees = list(cldf["TreeTable"])
    except (KeyError, OSError, ValueError):
        return []

    result = []
    for tree in trees:
        media = media_by_id.get(tree.get("Media_ID"))
        if not media:
            continue
        payload = _data_url_to_text(str(media.get("Download_URL") or ""))
        if payload:
            result.append(payload)
    return result


def _filesystem_tree_strings(
    dataset_path: Path,
    manifest: TemporalTreeManifest,
    dataset_name: str,
) -> list[str]:
    """Read explicitly configured and canonical CLDF Newick files."""
    relative_paths = [
        "cldf/tree.nwk",
        "cldf/tree.newick",
        "cldf/trees.nwk",
        "cldf/trees.newick",
        *manifest.paths_for(dataset_name),
    ]
    result = []
    seen_paths: set[Path] = set()
    for relative_path in relative_paths:
        path = Path(relative_path)
        if not path.is_absolute():
            path = dataset_path / path
        path = path.resolve()
        if path in seen_paths or not path.is_file():
            continue
        seen_paths.add(path)
        try:
            result.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            logger.warning("Could not decode temporal tree '%s'.", path)
    return result


def _parse_source_newick(newick_string: str) -> TreeNode:
    """Parse Newick after removing bracket comments used in source trees."""
    # Square-bracket comments are legal Newick annotations but are not
    # accepted by the lightweight parser used by the rest of this project.
    # They are descriptive here and do not carry topology or node identity.
    without_comments = re.sub(r"\[[^\[\]]*\]", "", newick_string).strip()
    return parse_newick(without_comments)


def discover_temporal_lineages(
    dataset: DatasetForms,
    manifest: TemporalTreeManifest,
) -> dict[str, dict[str, set[str]]]:
    """Extract historical target branches from authoritative source trees.

    An automatically derived target must be a *non-root internal node* whose
    label resolves to a historical-looking source variety.  Its direct child
    subtrees define the independent branches used to select inputs.
    """
    if dataset.source_path is None:
        return {}

    newick_strings = _cldf_tree_strings(dataset.source_path)
    newick_strings.extend(
        _filesystem_tree_strings(dataset.source_path, manifest, dataset.dataset_name)
    )
    resolver = _LanguageLabelResolver(dataset)
    result: dict[str, dict[str, set[str]]] = {}

    for tree_index, newick_string in enumerate(newick_strings):
        try:
            root = _parse_source_newick(newick_string)
        except (ValueError, IndexError):
            logger.warning(
                "Could not parse source Newick tree %d for dataset '%s'.",
                tree_index,
                dataset.dataset_name,
            )
            continue

        for node in _walk(root):
            if node.parent is None or node.is_leaf:
                continue
            target_id = resolver.resolve(node.label)
            if target_id is None or target_id not in dataset.forms_by_language:
                continue
            target = dataset.languages[target_id]
            if not (target.is_historical or _HISTORICAL_NAME.search(target.name)):
                continue

            branches: dict[str, set[str]] = {}
            for child_index, child in enumerate(node.children):
                descendants = {
                    descendant_id
                    for descendant in _walk(child)
                    if (descendant_id := resolver.resolve(descendant.label))
                    and descendant_id != target_id
                    and descendant_id in dataset.forms_by_language
                    and not dataset.languages[descendant_id].is_proto
                }
                if descendants:
                    branch_id = (
                        f"source_tree:{tree_index}:{target_id}:{child_index}"
                    )
                    branches[branch_id] = descendants
            if len(branches) >= 2:
                # Keep one authoritative topology per target.  A manually
                # curated lineage manifest overrides this later in generation.
                result.setdefault(target_id, branches)
    return result
