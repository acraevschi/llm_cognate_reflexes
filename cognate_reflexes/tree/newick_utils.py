"""Newick tree parsing, polytomy resolution, and tree traversal utilities.

This module is the algorithmic core of the tree layer.  It provides:

* A lightweight :class:`TreeNode` dataclass for representing phylogenetic
  trees in memory.
* Parsing from Newick strings (via the ``newick`` package) into
  :class:`TreeNode` trees.
* Exhaustive or sampled resolution of polytomies into binary trees.
* Post-order traversal yielding ``(left, right, parent)`` triplets.
* MRCA and distance computations.
"""

from __future__ import annotations

import copy
import itertools
import random
from dataclasses import dataclass, field
from typing import Iterator


# ======================================================================
# TreeNode
# ======================================================================


@dataclass
class TreeNode:
    """A node in a phylogenetic tree.

    Leaves carry a *label* (a Glottocode in the typical use-case).
    Internal nodes may be unlabelled (``label=None``), e.g. after
    polytomy resolution inserts new binary-split nodes.

    Attributes:
        label: Glottocode for leaf nodes, ``None`` for synthetic
            internal nodes.
        children: Direct descendants.
        branch_length: Edge weight from *parent* to this node.
            ``None`` when the tree is a pure classification tree
            (Glottolog trees never carry branch lengths).
        parent: Back-pointer to the parent node.  Not included in
            ``repr`` to avoid infinite recursion.
    """

    label: str | None = None
    children: list[TreeNode] = field(default_factory=list)
    branch_length: float | None = None
    parent: TreeNode | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """``True`` if this node has no children."""
        return len(self.children) == 0

    @property
    def is_polytomy(self) -> bool:
        """``True`` if this node has more than two children."""
        return len(self.children) > 2

    # ------------------------------------------------------------------
    # Leaf accessors
    # ------------------------------------------------------------------

    def get_leaves(self) -> list[TreeNode]:
        """Return all leaf nodes in the subtree rooted at this node.

        The leaves are returned in a pre-order (left-to-right) traversal
        order.
        """
        if self.is_leaf:
            return [self]
        leaves: list[TreeNode] = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves

    def get_leaf_labels(self) -> set[str]:
        """Return all leaf labels (Glottocodes) under this node.

        Labels that are ``None`` are silently excluded.
        """
        return {leaf.label for leaf in self.get_leaves() if leaf.label is not None}


# ======================================================================
# Parsing
# ======================================================================


def _convert_newick_node(nwk_node: object) -> TreeNode:
    """Recursively convert a ``newick.Node`` into a :class:`TreeNode`.

    The ``newick`` package represents parsed trees as ``newick.Node``
    objects.  This helper mirrors the topology into our own dataclass,
    setting parent back-pointers along the way.
    """
    node = TreeNode(
        label=nwk_node.name or None,  # type: ignore[union-attr]
        branch_length=nwk_node.length,  # type: ignore[union-attr]
    )
    for child in nwk_node.descendants:  # type: ignore[union-attr]
        child_node = _convert_newick_node(child)
        child_node.parent = node
        node.children.append(child_node)
    return node


def parse_newick(newick_str: str) -> TreeNode:
    """Parse a Newick string into a :class:`TreeNode` tree.

    Uses the ``newick`` package for the actual parsing, then converts
    the resulting tree into our own :class:`TreeNode` representation so
    that every node carries a *parent* back-pointer.

    Args:
        newick_str: A Newick-format string, e.g.
            ``"((A,B),(C,D));"``

    Returns:
        The root :class:`TreeNode`.

    Raises:
        ValueError: If the string does not contain exactly one tree.
    """
    import newick as nwk_pkg  # type: ignore[import-untyped]

    trees = nwk_pkg.loads(newick_str)
    if not trees:
        raise ValueError("Newick string did not contain any trees.")
    if len(trees) > 1:
        raise ValueError(
            f"Newick string contained {len(trees)} trees; expected exactly 1."
        )

    return _convert_newick_node(trees[0])


# ======================================================================
# Deep copy helper
# ======================================================================


def _deep_copy_tree(node: TreeNode) -> TreeNode:
    """Create a deep copy of a subtree, correctly wiring parent pointers.

    :func:`copy.deepcopy` would copy parent pointers too, leading to
    copies of parents-of-parents all the way to the root.  This helper
    only copies *downward*.
    """
    new_node = TreeNode(
        label=node.label,
        branch_length=node.branch_length,
    )
    for child in node.children:
        child_copy = _deep_copy_tree(child)
        child_copy.parent = new_node
        new_node.children.append(child_copy)
    return new_node


# ======================================================================
# Polytomy resolution
# ======================================================================


def _all_binary_trees(items: list[TreeNode]) -> list[TreeNode]:
    """Generate all full binary trees whose leaves are *items*.

    For *N* items the number of distinct binary topologies is the
    Catalan number C(N−1).  The algorithm works recursively:

    * N = 1 → return the single item (already a subtree).
    * N = 2 → return one node with both items as children.
    * N ≥ 3 → for every way to split *items* into two non-empty
      groups (avoiding mirror duplicates), recursively enumerate
      binary trees for each group, then combine via a new parent
      node.

    To avoid mirror-image duplicates, we fix the first element to
    always be in the "left" partition.

    .. warning:: This function enumerates **all** Catalan(N−1) trees.
       For N > ~10 this is extremely expensive.  Use
       :func:`_sample_random_binary_tree` when only a subset is needed.
    """
    n = len(items)

    if n == 1:
        return [_deep_copy_tree(items[0])]

    if n == 2:
        parent = TreeNode()
        left = _deep_copy_tree(items[0])
        right = _deep_copy_tree(items[1])
        left.parent = parent
        right.parent = parent
        parent.children = [left, right]
        return [parent]

    # N ≥ 3: enumerate all ways to split items into two non-empty groups.
    # Fix items[0] in the left group to avoid mirror duplicates.
    rest = items[1:]
    results: list[TreeNode] = []

    # For each non-empty proper subset S of rest, left = {items[0]} ∪ S,
    # right = rest \ S.  We iterate over subsets of size 0..len(rest)-1
    # for the *rest* elements assigned to left.  (size 0 means left has
    # only items[0]; size len(rest) is excluded so right is non-empty.)
    for r in range(0, len(rest)):
        for combo in itertools.combinations(range(len(rest)), r):
            left_indices = set(combo)
            left_items = [items[0]] + [rest[i] for i in range(len(rest)) if i in left_indices]
            right_items = [rest[i] for i in range(len(rest)) if i not in left_indices]

            for left_subtree in _all_binary_trees(left_items):
                for right_subtree in _all_binary_trees(right_items):
                    parent = TreeNode()
                    left_copy = left_subtree  # already a fresh copy
                    right_copy = right_subtree
                    left_copy.parent = parent
                    right_copy.parent = parent
                    parent.children = [left_copy, right_copy]
                    results.append(parent)

    return results


def _catalan_number(n: int) -> int:
    """Compute the *n*-th Catalan number: C(n) = (2n)! / ((n+1)! · n!).

    Used to decide whether exhaustive enumeration of binary tree
    topologies is feasible before allocating memory.
    """
    if n <= 1:
        return 1
    c = 1
    for i in range(n):
        c = c * 2 * (2 * i + 1) // (i + 2)
    return c


def _sample_random_binary_tree(
    items: list[TreeNode], rng: random.Random
) -> TreeNode:
    """Generate a single random binary tree whose leaves are *items*.

    Uses random partitioning to build one tree in O(N log N) time and
    O(N) space, avoiding the exponential cost of exhaustive enumeration.
    The distribution is not perfectly uniform over all Catalan topologies
    but provides good diversity for training-data generation.

    Args:
        items: Leaf nodes to combine.
        rng: Seeded random number generator.

    Returns:
        A new :class:`TreeNode` tree (deep copies of *items* as leaves).
    """
    n = len(items)

    if n == 1:
        return _deep_copy_tree(items[0])

    if n == 2:
        parent = TreeNode()
        left = _deep_copy_tree(items[0])
        right = _deep_copy_tree(items[1])
        left.parent = parent
        right.parent = parent
        parent.children = [left, right]
        return parent

    # Randomly partition items into two non-empty groups.
    shuffled = list(items)
    rng.shuffle(shuffled)
    split = rng.randint(1, n - 1)
    left_items = shuffled[:split]
    right_items = shuffled[split:]

    parent = TreeNode()
    left_subtree = _sample_random_binary_tree(left_items, rng)
    right_subtree = _sample_random_binary_tree(right_items, rng)
    left_subtree.parent = parent
    right_subtree.parent = parent
    parent.children = [left_subtree, right_subtree]
    return parent


def resolve_polytomy(
    node: TreeNode,
    max_resolutions: int = 50,
    rng: random.Random | None = None,
) -> list[TreeNode]:
    """Generate binary resolutions of a single polytomy node.

    For a node with *N* children the number of distinct fully-resolved
    binary topologies is the Catalan number *C(N−1)*.  When this count
    exceeds *max_resolutions*, random binary trees are **sampled
    directly** (O(N) per tree) instead of enumerating all C(N−1)
    topologies first.  This avoids the catastrophic memory usage that
    exhaustive enumeration causes for large polytomies (e.g. C(14) ≈
    2.7 million trees for a 15-child node).

    Each resolution is a **new subtree** (deep copy); the original tree
    is never mutated.  Newly inserted internal nodes have
    ``label=None``.

    Args:
        node: A :class:`TreeNode` that is a polytomy
            (``len(node.children) > 2``).  If the node is already
            binary or a leaf the function returns a single-element list
            containing a deep copy of *node*.
        max_resolutions: Maximum number of resolutions to return.
        rng: Seeded random number generator for reproducible sampling.
            If ``None``, a fresh unseeded ``random.Random()`` is used.

    Returns:
        A list of resolved subtrees, each a :class:`TreeNode` root
        whose leaf set is identical to *node*'s children.
    """
    if len(node.children) <= 2:
        return [_deep_copy_tree(node)]

    n_children = len(node.children)
    catalan = _catalan_number(n_children - 1)

    if catalan <= max_resolutions:
        # Small enough to enumerate exhaustively.
        all_trees = _all_binary_trees(node.children)
    else:
        # Too many topologies — sample random binary trees directly.
        if rng is None:
            rng = random.Random()
        all_trees = [
            _sample_random_binary_tree(node.children, rng)
            for _ in range(max_resolutions)
        ]

    # Transfer the original node's label and branch_length to each
    # resolution's root.
    for tree in all_trees:
        tree.label = node.label
        tree.branch_length = node.branch_length

    if len(all_trees) <= max_resolutions:
        return all_trees

    if rng is None:
        rng = random.Random()
    return rng.sample(all_trees, max_resolutions)


def _deep_copy_full_tree(root: TreeNode) -> TreeNode:
    """Deep-copy a complete tree (same as ``_deep_copy_tree``)."""
    return _deep_copy_tree(root)


def _collect_polytomies(node: TreeNode) -> list[TreeNode]:
    """Return all polytomy nodes in the subtree (pre-order)."""
    result: list[TreeNode] = []
    if node.is_polytomy:
        result.append(node)
    for child in node.children:
        result.extend(_collect_polytomies(child))
    return result


def _find_node_by_id(root: TreeNode, target_id: int) -> TreeNode | None:
    """Find a node in *root*'s subtree whose ``id()`` matches *target_id*.

    This is used after deep-copying to locate the copy of a node we
    wish to splice a resolved subtree into.
    """
    if id(root) == target_id:
        return root
    for child in root.children:
        found = _find_node_by_id(child, target_id)
        if found is not None:
            return found
    return None


def _replace_node(target: TreeNode, replacement: TreeNode) -> None:
    """Replace *target* with *replacement* in-place within *target*'s tree.

    *replacement*'s children and label are spliced into *target*.
    This modifies *target* in place (its parent's child list stays
    valid because the same object is reused).
    """
    target.label = replacement.label
    target.branch_length = replacement.branch_length
    target.children = replacement.children
    for child in target.children:
        child.parent = target


def resolve_all_polytomies(
    root: TreeNode,
    max_resolutions_per_node: int = 50,
    max_total_trees: int = 1000,
    rng: random.Random | None = None,
) -> list[TreeNode]:
    """Resolve **all** polytomies in a tree recursively bottom-up.

    Each polytomy is resolved in a bottom-up fashion to ensure that
    descendant resolutions are correctly preserved when resolving
    ancestor polytomies.

    Args:
        root: Root of the tree to resolve.
        max_resolutions_per_node: Cap per individual polytomy
            (passed through to :func:`resolve_polytomy`).
        max_total_trees: Hard cap on the total number of returned trees.
        rng: Seeded random number generator for reproducible sampling.

    Returns:
        A list of fully binary :class:`TreeNode` trees. Each tree is
        independent (its own deep copy).
    """
    if max_total_trees < 1:
        raise ValueError("max_total_trees must be positive")

    if root.is_leaf:
        return [_deep_copy_tree(root)]

    # 1. Resolve children first recursively
    resolved_children_combos = []
    for child in root.children:
        resolved_children_combos.append(
            resolve_all_polytomies(child, max_resolutions_per_node, max_total_trees, rng)
        )

    # 2. Compute a bounded sample of the Cartesian product.  Materialising
    # the full product before sampling can create millions of combinations
    # even when the returned result is capped to a few dozen trees.
    if rng is None:
        rng = random.Random()
    combo_count = 1
    for child_resolutions in resolved_children_combos:
        combo_count *= len(child_resolutions)

    if combo_count <= max_total_trees:
        child_combos = itertools.product(*resolved_children_combos)
    else:
        sampled_indices: set[tuple[int, ...]] = set()
        while len(sampled_indices) < max_total_trees:
            sampled_indices.add(
                tuple(rng.randrange(len(options)) for options in resolved_children_combos)
            )
        child_combos = (
            tuple(options[index] for options, index in zip(resolved_children_combos, indices))
            for indices in sorted(sampled_indices)
        )

    # 3. Resolve the current node if it is a polytomy
    resolved_roots: list[TreeNode] = []
    candidates_seen = 0
    for combo in child_combos:
        temp_root = TreeNode(label=root.label, branch_length=root.branch_length)
        # A child resolution can appear in several sampled combinations.  Each
        # output tree needs independent child objects and parent pointers.
        temp_root.children = [_deep_copy_tree(child) for child in combo]
        for c in temp_root.children:
            c.parent = temp_root

        if len(temp_root.children) > 2:
            candidates = resolve_polytomy(
                temp_root,
                max_resolutions=max_resolutions_per_node,
                rng=rng,
            )
        else:
            candidates = [temp_root]

        # Keep a bounded reservoir of full trees.  ``resolve_polytomy`` can
        # yield many candidates for one child combination, so capping only at
        # the end would still create a transient memory spike.
        for candidate in candidates:
            candidates_seen += 1
            if len(resolved_roots) < max_total_trees:
                resolved_roots.append(candidate)
            else:
                replacement = rng.randrange(candidates_seen)
                if replacement < max_total_trees:
                    resolved_roots[replacement] = candidate

    return resolved_roots


# ======================================================================
# Traversal
# ======================================================================


def postorder_triplets(
    root: TreeNode,
) -> Iterator[tuple[TreeNode, TreeNode, TreeNode]]:
    """Yield ``(left_child, right_child, parent)`` in post-order.

    The traversal visits leaves first and the root last, which is the
    natural order for bottom-up tree computations (e.g. Fitch or
    likelihood algorithms).

    This function **requires** a binary tree.  Call
    :func:`resolve_all_polytomies` first to ensure every internal node
    has exactly two children.

    Args:
        root: Root of a **binary** tree.

    Yields:
        3-tuples ``(left, right, parent)`` for every internal node,
        ordered bottom-up.

    Raises:
        ValueError: If any internal node has a number of children ≠ 2.
    """
    # Recurse into children first (post-order).
    if root.is_leaf:
        return

    if len(root.children) != 2:
        raise ValueError(
            f"postorder_triplets requires a binary tree, but node "
            f"{root.label!r} has {len(root.children)} children."
        )

    left, right = root.children

    yield from postorder_triplets(left)
    yield from postorder_triplets(right)
    yield (left, right, root)


# ======================================================================
# MRCA and distance
# ======================================================================


def _ancestor_path(node: TreeNode) -> list[TreeNode]:
    """Return the path from *node* up to (and including) the root."""
    path: list[TreeNode] = []
    current: TreeNode | None = node
    while current is not None:
        path.append(current)
        current = current.parent
    return path


def find_mrca(node_a: TreeNode, node_b: TreeNode) -> TreeNode | None:
    """Find the most recent common ancestor (MRCA) of two nodes.

    Args:
        node_a: First node.
        node_b: Second node.

    Returns:
        The deepest node that is an ancestor of both *node_a* and
        *node_b*, or ``None`` if the nodes do not share a common
        ancestor (i.e. they belong to different trees).
    """
    ancestors_a: set[int] = set()
    current: TreeNode | None = node_a
    while current is not None:
        ancestors_a.add(id(current))
        current = current.parent

    current = node_b
    while current is not None:
        if id(current) in ancestors_a:
            return current
        current = current.parent

    return None


def compute_distance(
    node_a: TreeNode,
    node_b: TreeNode,
    root: TreeNode,  # noqa: ARG001 — kept for API symmetry
) -> tuple[float | None, int]:
    """Compute the tree distance between two nodes.

    The distance is measured along the unique path through the tree
    (via the MRCA).

    Args:
        node_a: First node.
        node_b: Second node.
        root: Root of the tree (currently unused — MRCA is found via
            parent pointers — but kept for API symmetry and potential
            future validation).

    Returns:
        A 2-tuple ``(branch_length_distance, edge_count_distance)``.

        * *branch_length_distance* is the sum of branch lengths on the
          path from *node_a* to *node_b* through their MRCA.  It is
          ``None`` if **any** edge on the path lacks a branch length.
        * *edge_count_distance* is the number of edges on that path.

    Raises:
        ValueError: If the two nodes do not share a common ancestor.
    """
    mrca = find_mrca(node_a, node_b)
    if mrca is None:
        raise ValueError(
            f"Nodes {node_a.label!r} and {node_b.label!r} share no common ancestor."
        )

    def _path_to_mrca(node: TreeNode) -> tuple[float | None, int]:
        """Walk from *node* up to *mrca*, summing lengths and counting edges."""
        total_length: float = 0.0
        has_all_lengths = True
        edge_count = 0
        current: TreeNode | None = node
        while current is not None and current is not mrca:
            edge_count += 1
            if current.branch_length is not None:
                total_length += current.branch_length
            else:
                has_all_lengths = False
            current = current.parent
        return (total_length if has_all_lengths else None, edge_count)

    len_a, edges_a = _path_to_mrca(node_a)
    len_b, edges_b = _path_to_mrca(node_b)

    if len_a is not None and len_b is not None:
        bl_dist: float | None = len_a + len_b
    else:
        bl_dist = None

    return (bl_dist, edges_a + edges_b)


def tree_depth(node: TreeNode) -> int:
    """Return the depth of *node* (number of edges from the root).

    The root has depth 0.

    Args:
        node: Node whose depth to compute.

    Returns:
        Non-negative integer depth.
    """
    depth = 0
    current: TreeNode | None = node.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth
