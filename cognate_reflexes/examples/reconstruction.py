"""Stage 2: Proto-language reconstruction example generation.

Generate training examples where up to 5 *attested descendant*
languages serve as inputs and a *proto-language* with real attested data
serves as the target.  All target forms are masked — the model must
reconstruct the ancestral vocabulary.
"""

from __future__ import annotations

import logging
import random
import itertools
from typing import Iterator

from cognate_reflexes.config import Config
from cognate_reflexes.data.loader import DatasetForms
from cognate_reflexes.tree.newick_utils import (
    TreeNode,
    tree_depth,
)
from cognate_reflexes.examples.models import (
    Form,
    LanguageData,
    TrainingExample,
    ExampleMetadata,
)

logger = logging.getLogger(__name__)


def _has_language_data(
    dataset: DatasetForms,
    glottocode: str | None,
) -> bool:
    """Check whether *glottocode* has forms in *dataset*."""
    if glottocode is None:
        return False
    return glottocode in dataset.forms_by_language


def _shared_cognate_sets_nary(
    dataset: DatasetForms,
    gcs: list[str],
) -> list[str]:
    """Find cognate sets shared across all languages in *gcs*."""
    if not gcs:
        return []
    shared = set(dataset.forms_by_language.get(gcs[0], {}).keys())
    for gc in gcs[1:]:
        shared &= set(dataset.forms_by_language.get(gc, {}).keys())
    return sorted(shared)


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


def _get_attested_frontier(
    dataset: DatasetForms,
    node: TreeNode,
) -> list[TreeNode]:
    """Find the first attested descendants down each branch from *node*."""
    if _has_language_data(dataset, node.label):
        return [node]
    frontier = []
    for child in node.children:
        frontier.extend(_get_attested_frontier(dataset, child))
    return frontier


def generate_reconstruction_examples(
    dataset: DatasetForms,
    tree: TreeNode,
    config: Config,
    rng: random.Random | None = None,
) -> Iterator[TrainingExample]:
    """Generate proto-language reconstruction examples using a strict triplet format.

    For each internal node whose label corresponds to a proto-language
    in the dataset:

    1. Since the tree is binary, it has exactly a left and a right child.
    2. Find the attested frontier for the left child and sample 1 language.
    3. Find the attested frontier for the right child and sample 1 language.
    4. If both exist, use them as the 2 inputs.
    5. Find shared cognate sets across the 2 inputs and the target.
    6. Mask all target forms (full reconstruction).
    7. Yield the ``TrainingExample``.

    Args:
        dataset: Loaded forms from a single CLDF dataset.
        tree: Binary tree (e.g., from resolve_all_polytomies).
        config: Pipeline configuration.
        rng: Random number generator for reproducibility.

    Yields:
        :class:`TrainingExample` objects with ``task="reconstruction"``.
    """
    rng = rng or random.Random()

    if not dataset.proto_languages:
        logger.debug(
            "Dataset '%s': no proto-languages found — skipping "
            "reconstruction example generation.",
            dataset.dataset_name,
        )
        return

    # Collect all proto-language nodes in the tree
    proto_nodes = []
    def _collect(n: TreeNode) -> None:
        if _has_language_data(dataset, n.label) and n.label in dataset.proto_languages:
            proto_nodes.append(n)
        for child in n.children:
            _collect(child)
    _collect(tree)

    for parent in proto_nodes:
        parent_gc = parent.label
        if parent_gc is None:
            continue
            
        if len(parent.children) != 2:
            continue
            
        left_child, right_child = parent.children
        left_frontier = _get_attested_frontier(dataset, left_child)
        right_frontier = _get_attested_frontier(dataset, right_child)
        
        if not left_frontier or not right_frontier:
            continue
            
        unique_left = list({n.label: n for n in left_frontier if n.label}.values())
        unique_right = list({n.label: n for n in right_frontier if n.label}.values())
        
        # Generate multiple combinations of left and right frontier leaves
        pairs = list(itertools.product(unique_left, unique_right))
        rng.shuffle(pairs)

        max_pairs = 20
        for left_leaf, right_leaf in pairs[:max_pairs]:
            perm_nodes = [left_leaf, right_leaf]
            
            input_gcs = [n.label for n in perm_nodes if n.label]
            all_gcs = input_gcs + [parent_gc]
            
            shared_cogsets = _shared_cognate_sets_nary(dataset, all_gcs)
            if len(shared_cogsets) < config.min_cognates:
                continue
                
            if len(shared_cogsets) > config.max_cognates:
                shared_cogsets = sorted(
                    rng.sample(shared_cogsets, config.max_cognates)
                )
                
            shuffled_cogsets = list(shared_cogsets)
            rng.shuffle(shuffled_cogsets)
            
            inputs_data = [
                _build_language_data(dataset, gc, shuffled_cogsets, rng)
                for gc in input_gcs
            ]
            target_data = _build_language_data(dataset, parent_gc, shuffled_cogsets, rng)
            masked_indices = list(range(len(target_data.forms)))
            
            glottocodes = tuple(input_gcs + [parent_gc])
            mrca_depth = tree_depth(parent)
            
            branch_lengths = []
            for n in perm_nodes:
                bl, _ = _path_to_node(n, parent)
                branch_lengths.append(bl)
                
            coordinates: dict[str, tuple[float, float] | None] = {}
            for variety_id in glottocodes:
                lang = dataset.languages.get(variety_id)
                if lang and lang.latitude is not None and lang.longitude is not None:
                    coordinates[variety_id] = (lang.latitude, lang.longitude)
                else:
                    coordinates[variety_id] = None

            concept_ids = []
            for cid in shuffled_cogsets:
                found = False
                for variety_id in glottocodes:
                    forms = dataset.forms_by_language.get(variety_id, {}).get(cid, [])
                    if forms and forms[0].concepticon_id:
                        concept_ids.append(forms[0].concepticon_id)
                        found = True
                        break
                if not found:
                    concept_ids.append("")

            metadata = ExampleMetadata(
                source_dataset=dataset.dataset_name,
                language_family=dataset.family or "unknown",
                tree_depth=mrca_depth,
                branch_lengths=branch_lengths,
                num_cognate_sets=len(shuffled_cogsets),
                glottocodes=tuple(
                    dataset.languages[variety_id].glottocode
                    for variety_id in glottocodes
                ),
                variety_ids=glottocodes,
                coordinates=coordinates,
                concept_ids=concept_ids,
                cognateset_ids=shuffled_cogsets,
                target_kind="proto",
            )

            yield TrainingExample(
                task="reconstruction",
                inputs=inputs_data,
                target=target_data,
                masked_indices=masked_indices,
                metadata=metadata,
            )
