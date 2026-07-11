"""Stage 1: Cognate reflex example generation.

Generate training examples from groups of *sister* (leaf) languages
that share cognate sets. Up to 5 languages serve as inputs and one
as the target whose forms are partially masked. The model must
predict the masked forms.
"""

from __future__ import annotations

import itertools
import logging
import random
from typing import Iterator
from collections import defaultdict

from cognate_reflexes.config import Config
from cognate_reflexes.data.loader import DatasetForms
from cognate_reflexes.tree.newick_utils import (
    TreeNode,
    compute_distance,
    find_mrca,
    tree_depth,
)
from cognate_reflexes.examples.masking import apply_masking
from cognate_reflexes.examples.models import (
    Form,
    LanguageData,
    TrainingExample,
    ExampleMetadata,
)

logger = logging.getLogger(__name__)


def _get_leaf_node_map(tree: TreeNode) -> dict[str, TreeNode]:
    """Build a mapping from leaf label (Glottocode) to TreeNode."""
    return {leaf.label: leaf for leaf in tree.get_leaves() if leaf.label}


def _shared_cognate_sets_nary(
    dataset: DatasetForms,
    glottocodes: tuple[str, ...],
) -> list[str]:
    """Find cognate set IDs shared across all languages.

    A cognate set is "shared" if all languages have at least one
    form in that cognate set.
    """
    if not glottocodes:
        return []

    sets_per_lang = []
    for gc in glottocodes:
        lang_forms = dataset.forms_by_language.get(gc, {})
        sets_per_lang.append(set(lang_forms.keys()))

    shared = sets_per_lang[0]
    for s in sets_per_lang[1:]:
        shared = shared & s

    return sorted(shared)


def _check_distance_ok(
    nodes: list[TreeNode],
    root: TreeNode,
    max_branch_length: float | None,
    max_edge_distance: int | None,
) -> bool:
    """Check if the maximum pairwise distance is within bounds."""
    if max_branch_length is None and max_edge_distance is None:
        return True

    for a, b in itertools.combinations(nodes, 2):
        bl_dist, edge_dist = compute_distance(a, b, root)

        if max_branch_length is not None and bl_dist is not None:
            if bl_dist > max_branch_length:
                return False
        elif max_edge_distance is not None:
            # Fall back to edge count when branch lengths unavailable
            if edge_dist > max_edge_distance:
                return False

    return True


def _pick_one_form_per_cogset(
    forms_by_cogset: dict[str, list[Form]],
    cogset_ids: list[str],
    rng: random.Random,
) -> list[Form]:
    """For each cognate set, randomly pick one form."""
    result = []
    for cid in cogset_ids:
        forms = forms_by_cogset.get(cid, [])
        if forms:
            result.append(rng.choice(forms))
    return result


def _build_language_data(
    dataset: DatasetForms,
    glottocode: str,
    cogset_ids: list[str],
    rng: random.Random,
) -> LanguageData:
    """Build a LanguageData object with one form per cognate set."""
    lang_meta = dataset.languages.get(glottocode)
    forms_by_cogset = dataset.forms_by_language.get(glottocode, {})
    forms = _pick_one_form_per_cogset(forms_by_cogset, cogset_ids, rng)

    return LanguageData(
        glottocode=glottocode,
        name=lang_meta.name if lang_meta else glottocode,
        forms=forms,
        latitude=lang_meta.latitude if lang_meta else None,
        longitude=lang_meta.longitude if lang_meta else None,
        family=lang_meta.family if lang_meta else None,
        is_proto=lang_meta.is_proto if lang_meta else False,
        variety_id=lang_meta.identifier if lang_meta else glottocode,
        tree_glottocode=lang_meta.tree_glottocode if lang_meta else None,
        is_historical=lang_meta.is_historical if lang_meta else False,
        date_before_present=(
            lang_meta.date_before_present if lang_meta else None
        ),
        clade_path=lang_meta.clade_path if lang_meta else (),
    )


def _path_to_node(
    start: TreeNode, target: TreeNode
) -> tuple[float | None, int]:
    """Walk from *start* up to *target*, summing branch lengths."""
    total_length: float = 0.0
    has_all_lengths = True
    edge_count = 0
    current: TreeNode | None = start
    while current is not None and current is not target:
        edge_count += 1
        if current.branch_length is not None:
            total_length += current.branch_length
        else:
            has_all_lengths = False
        current = current.parent
    return (total_length if has_all_lengths else None, edge_count)


def _build_metadata(
    dataset: DatasetForms,
    glottocodes: tuple[str, ...],
    cogset_ids: list[str],
    tree: TreeNode,
    leaf_nodes: dict[str, TreeNode],
) -> ExampleMetadata:
    """Build ExampleMetadata for an example."""
    nodes = [leaf_nodes.get(gc) for gc in glottocodes]
    valid_nodes = [n for n in nodes if n is not None]

    mrca_depth = 0
    branch_lengths = [None] * (len(glottocodes) - 1)

    if len(valid_nodes) >= 2:
        mrca = valid_nodes[0]
        for n in valid_nodes[1:]:
            mrca_next = find_mrca(mrca, n)
            if mrca_next:
                mrca = mrca_next
        
        if mrca:
            mrca_depth = tree_depth(mrca)
            for i, gc in enumerate(glottocodes[:-1]):  # exclude target
                node = leaf_nodes.get(gc)
                if node:
                    bl, _ = _path_to_node(node, mrca)
                    branch_lengths[i] = bl

    # Coordinates
    coordinates: dict[str, tuple[float, float] | None] = {}
    for variety_id in glottocodes:
        lang = dataset.languages.get(variety_id)
        if lang and lang.latitude is not None and lang.longitude is not None:
            coordinates[variety_id] = (lang.latitude, lang.longitude)
        else:
            coordinates[variety_id] = None

    # Concept IDs
    concept_ids = []
    for cid in cogset_ids:
        # Get concept from any language's forms for this cogset
        found = False
        for variety_id in glottocodes:
            forms = dataset.forms_by_language.get(variety_id, {}).get(cid, [])
            if forms and forms[0].concepticon_id:
                concept_ids.append(forms[0].concepticon_id)
                found = True
                break
        if not found:
            concept_ids.append("")

    return ExampleMetadata(
        source_dataset=dataset.dataset_name,
        language_family=dataset.family or "unknown",
        tree_depth=mrca_depth,
        branch_lengths=branch_lengths,
        num_cognate_sets=len(cogset_ids),
        glottocodes=tuple(
            dataset.languages[variety_id].glottocode
            for variety_id in glottocodes
        ),
        variety_ids=glottocodes,
        coordinates=coordinates,
        concept_ids=concept_ids,
        cognateset_ids=cogset_ids,
        target_kind="reflex",
    )


def _sample_valid_leaf(node: TreeNode, valid_leaves: dict[str, TreeNode], rng: random.Random) -> TreeNode | None:
    """Sample one valid leaf from the subtree rooted at *node*."""
    leaves = [leaf for leaf in node.get_leaves() if leaf.label in valid_leaves]
    if not leaves:
        return None
    return rng.choice(leaves)


def generate_cognate_reflex_examples(
    dataset: DatasetForms,
    tree: TreeNode,
    config: Config,
    rng: random.Random | None = None,
) -> Iterator[TrainingExample]:
    """Generate cognate reflex examples using a strict triplet format.

    For each valid leaf node (target):
    1. Find its immediate sister branch and sample 1 valid leaf.
    2. Find its aunt branch (parent's sister) and sample 1 valid leaf.
    3. If both exist, use them as the 2 inputs for reconstructing the target.
    4. Apply random masking to the target forms.
    5. Yield the ``TrainingExample``.

    This assumes the input tree is strictly binary (e.g., via resolve_all_polytomies).

    Args:
        dataset: Loaded forms from a single CLDF dataset.
        tree: Binary tree containing only languages in this dataset.
        config: Pipeline configuration.
        rng: Random number generator for reproducibility.

    Yields:
        :class:`TrainingExample` objects with ``task="cognate_reflex"``.
    """
    rng = rng or random.Random()

    leaf_nodes = _get_leaf_node_map(tree)

    # Only consider leaves that are in both the tree AND the dataset,
    # and that are NOT proto-languages.
    valid_leaves = {
        gc: node
        for gc, node in leaf_nodes.items()
        if gc in dataset.forms_by_language
        and gc not in dataset.proto_languages
    }

    if len(valid_leaves) < 2:
        logger.debug(
            "Dataset '%s': fewer than 2 valid leaf languages — skipping "
            "cognate reflex generation.",
            dataset.dataset_name,
        )
        return

    for target_gc, node in valid_leaves.items():
        if node.parent is None or len(node.parent.children) != 2:
            continue
        
        sister_branch = [c for c in node.parent.children if c is not node][0]
        
        grandparent = node.parent.parent
        if grandparent is None or len(grandparent.children) != 2:
            continue
            
        aunt_branch = [c for c in grandparent.children if c is not node.parent][0]
        
        sister_leaf = _sample_valid_leaf(sister_branch, valid_leaves, rng)
        aunt_leaf = _sample_valid_leaf(aunt_branch, valid_leaves, rng)
        
        if sister_leaf is None or aunt_leaf is None:
            continue
            
        input_gcs = [sister_leaf.label, aunt_leaf.label]
        all_gcs = tuple(input_gcs + [target_gc])
        
        nodes = [sister_leaf, aunt_leaf, node]
        if not _check_distance_ok(
            nodes, tree, config.max_branch_length, config.max_edge_distance
        ):
            continue

        shared_cogsets = _shared_cognate_sets_nary(dataset, all_gcs)
        if len(shared_cogsets) < config.min_cognates:
            continue

        if len(shared_cogsets) > config.max_cognates:
            shared_cogsets = sorted(rng.sample(shared_cogsets, config.max_cognates))

        shuffled_cogsets = list(shared_cogsets)
        rng.shuffle(shuffled_cogsets)

        inputs_data = [
            _build_language_data(dataset, gc, shuffled_cogsets, rng)
            for gc in input_gcs
            if gc is not None
        ]
        target_data = _build_language_data(dataset, target_gc, shuffled_cogsets, rng)

        masked_indices = apply_masking(
            num_forms=len(target_data.forms),
            mask_ratio=config.mask_ratio,
            rng=rng,
        )

        metadata = _build_metadata(
            dataset, all_gcs, shuffled_cogsets, tree, leaf_nodes
        )

        yield TrainingExample(
            task="cognate_reflex",
            inputs=inputs_data,
            target=target_data,
            masked_indices=masked_indices,
            metadata=metadata,
        )
