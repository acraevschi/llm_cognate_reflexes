from __future__ import annotations

from pathlib import Path
import random

import pytest

from cognate_reflexes import Config, ExampleGenerator
from cognate_reflexes.tree.newick_utils import TreeNode, resolve_all_polytomies


def _leaf(label: str) -> TreeNode:
    return TreeNode(label=label)


def _set_parents(node: TreeNode) -> None:
    for child in node.children:
        child.parent = node
        _set_parents(child)


def _walk(node: TreeNode):
    yield node
    for child in node.children:
        yield from _walk(child)


def test_polytomy_resolution_respects_global_tree_cap() -> None:
    root = TreeNode(
        children=[
            TreeNode(children=[_leaf("a"), _leaf("b"), _leaf("c")]),
            TreeNode(children=[_leaf("d"), _leaf("e"), _leaf("f")]),
        ]
    )
    _set_parents(root)

    trees = resolve_all_polytomies(
        root,
        max_resolutions_per_node=8,
        max_total_trees=3,
        rng=random.Random(42),
    )

    assert len(trees) == 3
    for tree in trees:
        for node in _walk(tree):
            assert len(node.children) in {0, 2}
            assert all(child.parent is node for child in node.children)


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "data/lexibank/iecor").exists(),
    reason="requires the local IE-CoR clone",
)
def test_parallel_generation_streams_iecor_in_bounded_batches(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    (tmp_path / "iecor").symlink_to(
        project_root / "data/lexibank/iecor",
        target_is_directory=True,
    )
    config = Config(
        data_dir=str(tmp_path),
        glottolog_dir=str(project_root / "data/glottolog"),
        task="reconstruction",
        min_cognates=10,
        max_cognates=120,
        max_triplets_per_dataset=12,
        historical_lineages_path=str(project_root / "data/historical_lineages.csv"),
        temporal_trees_path=str(project_root / "data/temporal_trees.csv"),
    )

    # IE-CoR emits 12 examples, so the worker sends more than one 8-item
    # batch instead of materialising its entire result in the parent process.
    examples = list(ExampleGenerator(config).generate(workers=2))

    assert len(examples) == 12
    assert {example.metadata.target_kind for example in examples} == {"historical"}
