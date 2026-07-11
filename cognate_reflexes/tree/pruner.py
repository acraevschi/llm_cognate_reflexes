"""Tree pruning: restrict a phylogenetic tree to a subset of leaves."""

from __future__ import annotations

from cognate_reflexes.tree.newick_utils import TreeNode, _deep_copy_tree


def collapse_single_children(node: TreeNode, keep_labels: set[str] | None = None) -> TreeNode:
    """Collapse chains of single-child internal nodes.

    If an internal node has exactly one child, the child is promoted to
    take the parent's place.  Branch lengths are summed when both the
    parent and child carry values; if either is ``None`` the resulting
    edge also gets ``None``.

    The transformation is applied recursively in a bottom-up fashion so
    that multi-link chains (A→B→C→D) are fully collapsed in a single
    pass.

    Args:
        node: Root of the (sub)tree to collapse.  **Modified in place.**
        keep_labels: Optional set of labels to preserve (do not collapse).

    Returns:
        The (possibly different) root node after collapsing.

    Example::

        #  A ─── B ─── C   →   A ─── C  (branch_length = sum)
    """
    keep_labels = keep_labels or set()
    # First, recurse into children so that deeper chains are already
    # collapsed when we inspect this level.
    node.children = [collapse_single_children(child, keep_labels) for child in node.children]
    for child in node.children:
        child.parent = node

    # Collapse: while this node has exactly one child, absorb it.
    # But do NOT collapse if this node's label is in keep_labels.
    while len(node.children) == 1 and (node.label not in keep_labels):
        only_child = node.children[0]

        # Sum branch lengths.
        if node.branch_length is not None and only_child.branch_length is not None:
            only_child.branch_length = node.branch_length + only_child.branch_length
        elif node.branch_length is not None:
            only_child.branch_length = node.branch_length
        # else: keep only_child.branch_length as-is (could be None)

        # Promote the child.
        only_child.parent = node.parent
        node = only_child

    return node


def prune_tree(root: TreeNode, keep_labels: set[str]) -> TreeNode | None:
    """Prune a tree so that only leaves with labels in *keep_labels* remain.

    The function operates on a **deep copy** of the input tree — the
    original is never modified.

    After removing unwanted leaves, internal nodes that become leaves
    (i.e. lose all descendants) are also removed, and single-child
    chains are collapsed via :func:`collapse_single_children`.

    Args:
        root: Root of the tree to prune.
        keep_labels: Set of leaf labels (typically Glottocodes) to
            retain.  Leaves whose label is ``None`` are always removed
            unless ``None`` is explicitly in *keep_labels* (unusual).

    Returns:
        The root of the pruned tree, or ``None`` if no leaves survive
        the pruning.

    Example::

        >>> from cognate_reflexes.tree.newick_utils import parse_newick
        >>> tree = parse_newick("((A,B),(C,D));")
        >>> pruned = prune_tree(tree, {"A", "C"})
        >>> sorted(pruned.get_leaf_labels())
        ['A', 'C']
    """
    # Work on a deep copy so the caller's tree is untouched.
    tree = _deep_copy_tree(root)

    pruned = _prune_recursive(tree, keep_labels)
    if pruned is None:
        return None

    pruned = collapse_single_children(pruned, keep_labels)

    # If the root ended up as a single internal node with no meaningful
    # topology (one child), that was already handled by collapse.
    # Ensure the returned root has no parent pointer.
    pruned.parent = None
    return pruned


def _prune_recursive(node: TreeNode, keep_labels: set[str]) -> TreeNode | None:
    """Recursively remove leaves not in *keep_labels*.

    Returns ``None`` when the entire subtree should be removed.
    """
    surviving_children: list[TreeNode] = []
    for child in node.children:
        result = _prune_recursive(child, keep_labels)
        if result is not None:
            result.parent = node
            surviving_children.append(result)

    node.children = surviving_children

    # Keep this node if it is in keep_labels OR if it has surviving children
    if (node.label in keep_labels) or surviving_children:
        return node
    return None
