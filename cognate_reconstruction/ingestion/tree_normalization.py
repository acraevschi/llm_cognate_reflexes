"""Topology-preserving pruning and unary-node collapse for input trees."""

from __future__ import annotations

import math
import re

from cognate_reflexes.tree.newick_utils import TreeNode

_UNQUOTED_LABEL = re.compile(r"^[A-Za-z0-9_.-]+$")


def _combine_lengths(upper: float | None, lower: float | None) -> float | None:
    if upper is None and lower is None:
        return None
    if upper is None or lower is None:
        return None
    return upper + lower


def normalize_tree(root: TreeNode, usable_leaf_ids: set[str]) -> TreeNode:
    """Prune unusable leaf branches and collapse every resulting unary node."""
    if len(usable_leaf_ids) < 2:
        raise ValueError("tree normalization requires at least two usable lexicons")

    def visit(node: TreeNode) -> TreeNode | None:
        if node.is_leaf:
            if node.label not in usable_leaf_ids:
                return None
            return TreeNode(label=node.label, branch_length=node.branch_length)

        children = [child for child in (visit(item) for item in node.children) if child]
        if not children:
            return None
        if len(children) == 1:
            child = children[0]
            child.branch_length = _combine_lengths(node.branch_length, child.branch_length)
            return child
        normalized = TreeNode(
            label=node.label,
            branch_length=node.branch_length,
            children=children,
        )
        for child in children:
            child.parent = normalized
        return normalized

    normalized_root = visit(root)
    if normalized_root is None:
        raise ValueError("tree normalization removed every branch")
    actual = normalized_root.get_leaf_labels()
    missing = sorted(usable_leaf_ids - actual)
    if missing:
        raise ValueError(f"usable lexicons missing from tree: {missing}")
    if normalized_root.is_leaf:
        raise ValueError("tree normalization left fewer than two usable branches")
    normalized_root.parent = None
    normalized_root.branch_length = None
    return normalized_root


def _label(value: str | None) -> str:
    if value is None:
        return ""
    if _UNQUOTED_LABEL.fullmatch(value):
        return value
    return "'" + value.replace("'", "''") + "'"


def _length(value: float | None) -> str:
    # The ``newick`` package represents omitted lengths as 0.0, so omit zero
    # here to preserve classification-style Glottolog trees on round-trip.
    if value is None or value == 0.0:
        return ""
    if not math.isfinite(value) or value < 0:
        raise ValueError("Newick branch lengths must be finite and non-negative")
    return f":{value:g}"


def to_newick(root: TreeNode) -> str:
    """Serialize a normalized tree without resolving its polytomies."""
    def render(node: TreeNode) -> str:
        descendants = ""
        if node.children:
            descendants = "(" + ",".join(render(child) for child in node.children) + ")"
        return descendants + _label(node.label) + _length(node.branch_length)

    return render(root) + ";"
