"""Main ExampleGenerator API — the primary entry point for the package.

Ties together data loading, tree management, and example generation into
a single convenient class that can be used from a fine-tuning script::

    from cognate_reflexes import ExampleGenerator, Config

    gen = ExampleGenerator(config=Config(task="cognate_reflex"))
    for example in gen.generate():
        print(example)
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterator

from cognate_reflexes.config import Config
from cognate_reflexes.data.loader import CLDFLoader, DatasetForms
from cognate_reflexes.data.registry import DatasetRegistry
from cognate_reflexes.tree.glottolog import GlottologTree
from cognate_reflexes.tree.newick_utils import (
    TreeNode,
    parse_newick,
    resolve_all_polytomies,
    find_mrca,
)
from cognate_reflexes.tree.pruner import prune_tree
from cognate_reflexes.examples.cognate_reflex import (
    generate_cognate_reflex_examples,
)
from cognate_reflexes.examples.models import TrainingExample
from cognate_reflexes.examples.reconstruction import (
    generate_reconstruction_examples,
)

logger = logging.getLogger(__name__)


import hashlib

def _process_dataset_worker(dataset_path: str, config: Config) -> list[TrainingExample]:
    """Worker function for multiprocessing."""
    gen = ExampleGenerator(config)
    
    # Create a deterministic but unique seed for this dataset to avoid correlated
    # randomness across worker processes.
    path_hash = int(hashlib.md5(str(dataset_path).encode('utf-8')).hexdigest(), 16)
    dataset_seed = (config.seed + path_hash) % (2**32)
    gen._rng = random.Random(dataset_seed)
    
    ds = gen.loader.load_dataset(dataset_path)
    if not ds:
        return []
    return list(gen._generate_for_dataset(ds))


class ExampleGenerator:
    """Main API for generating training examples from Lexibank data.

    Usage::

        gen = ExampleGenerator(
            data_dir="./data/lexibank",
            glottolog_dir="./data/glottolog",
            task="cognate_reflex",
        )
        for example in gen.generate():
            print(example)

        # Or materialise to disk
        gen.materialize("output/examples.jsonl")

    Args:
        config: Configuration object.  If ``None``, one is created from
            ``**kwargs``.
        **kwargs: Override individual :class:`Config` fields when
            *config* is ``None``.
    """

    def __init__(self, config: Config | None = None, **kwargs: object) -> None:
        if config is None:
            config = Config(**kwargs)  # type: ignore[arg-type]
        self.config = config
        self._rng = random.Random(config.seed)

        # Lazily initialised components.
        self._loader: CLDFLoader | None = None
        self._glottolog: GlottologTree | None = None
        self._registry: DatasetRegistry | None = None
        self._proto_name_map_cached: dict[str, str] | None = None

    @property
    def proto_name_map(self) -> dict[str, str]:
        """Lazy index of Glottolog names/clean names to Glottocodes."""
        if self._proto_name_map_cached is None:
            import re
            name_map = {}
            for lang in self.glottolog.api.languoids():
                norm = lang.name.lower().strip()
                name_map.setdefault(norm, lang.id)
                clean = re.sub(r'^[Pp]roto[- ]', '', norm).replace('*', '').strip()
                if clean != norm:
                    name_map.setdefault(clean, lang.id)
                # Suffix-stripped
                for suffix in ['an', 'ic', 'ish', 'ese', 'oid', 'oc']:
                    if clean.endswith(suffix):
                        stripped = clean[:-len(suffix)]
                        name_map.setdefault(stripped, lang.id)
            self._proto_name_map_cached = name_map
        return self._proto_name_map_cached

    # ------------------------------------------------------------------
    # Lazy property accessors
    # ------------------------------------------------------------------

    @property
    def loader(self) -> CLDFLoader:
        """Cached CLDF loader."""
        if self._loader is None:
            self._loader = CLDFLoader(self.config.data_dir)
        return self._loader

    @property
    def glottolog(self) -> GlottologTree:
        """Cached Glottolog tree interface."""
        if self._glottolog is None:
            self._glottolog = GlottologTree(self.config.glottolog_dir)
        return self._glottolog

    @property
    def registry(self) -> DatasetRegistry:
        """Cached dataset registry (scanned on first access)."""
        if self._registry is None:
            self._registry = DatasetRegistry(self.config.data_dir)
            self._registry.scan()
        return self._registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, workers: int = 1) -> Iterator[TrainingExample]:
        """Generate examples lazily across all suitable datasets.

        For each dataset that matches the current task requirements:

        1. Load forms via :class:`CLDFLoader`.
        2. Determine the language family / families present.
        3. Get Glottolog tree(s) and prune.
        4. Delegate to the task-specific generator.

        Args:
            workers: Number of parallel workers to use. If > 1, uses
                ProcessPoolExecutor.

        Yields:
            :class:`TrainingExample` objects.
        """
        datasets = self._discover_datasets()

        if workers <= 1:
            for ds_info in datasets:
                ds = self.loader.load_dataset(ds_info.path)
                if ds is None:
                    continue

                logger.info(
                    "Processing dataset '%s' (%d languages, %d cognate sets).",
                    ds.dataset_name,
                    ds.num_languages,
                    ds.num_cognate_sets,
                )

                yield from self._generate_for_dataset(ds)
        else:
            import concurrent.futures
            paths = [ds_info.path for ds_info in datasets]
            logger.info("Starting ProcessPoolExecutor with %d workers.", workers)
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_dataset_worker, path, self.config): path 
                    for path in paths
                }
                for future in concurrent.futures.as_completed(futures):
                    path = futures[future]
                    try:
                        examples = future.result()
                        logger.info("Finished dataset %s: %d examples.", path, len(examples))
                        yield from examples
                    except Exception:
                        logger.exception("Worker failed processing dataset '%s'", path)

    def materialize(self, path: str, format: str = "jsonl") -> int:
        """Write all examples to disk.

        Args:
            path: Output file path.
            format: Output format (currently only ``"jsonl"``).

        Returns:
            Number of examples written.
        """
        from cognate_reflexes.formatting.serializer import ExampleSerializer

        serializer = ExampleSerializer()
        return serializer.write_jsonl(self.generate(), path)

    def stats(self) -> dict[str, object]:
        """Return summary statistics about available data.

        Returns:
            Dictionary with keys like ``num_datasets``,
            ``num_datasets_with_cognates``, ``families``, etc.
        """
        all_ds = self.registry.list_all()
        with_cognates = self.registry.filter(has_cognates=True)
        with_proto = self.registry.filter(has_proto_forms=True)

        families: set[str] = set()
        total_langs = 0
        total_forms = 0
        for ds in all_ds:
            families.update(ds.families)
            total_langs += ds.num_languages
            total_forms += ds.num_forms

        return {
            "num_datasets": len(all_ds),
            "num_datasets_with_cognates": len(with_cognates),
            "num_datasets_with_proto": len(with_proto),
            "families": sorted(families),
            "total_languages": total_langs,
            "total_forms": total_forms,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_datasets(self) -> list:
        """Get the list of datasets to process based on the task."""
        if self.config.task == "reconstruction":
            return self.registry.filter(
                has_cognates=True, has_proto_forms=True
            )
        else:
            return self.registry.filter(has_cognates=True)

    def _generate_for_dataset(
        self, dataset: DatasetForms
    ) -> Iterator[TrainingExample]:
        """Generate training examples for a single dataset."""
        dataset_examples: list[TrainingExample] = []
        seen_hashes: set[int] = set()
        count = 0
        max_triplets = self.config.max_triplets_per_dataset

        # Clean proto name helper
        import re
        def clean_proto_name(name: str) -> str:
            name = name.lower()
            name = re.sub(r'^[Pp]roto[- ]', '', name)
            name = name.replace('*', '').strip()
            return name

        # Group languages by family.
        # First, resolve missing family names for proto-languages based on their descendants.
        for proto_gc in dataset.proto_languages:
            proto_lang = dataset.languages[proto_gc]
            if not proto_lang.family or proto_lang.family.lower() == "unknown":
                attested_langs = [
                    lang for lang in dataset.languages.values()
                    if not lang.is_proto
                ]
                clean_p = clean_proto_name(proto_lang.name)
                p_subgroup = (proto_lang.subgroup or "").strip()
                
                descendants = []
                if p_subgroup:
                    descendants = [
                        lang for lang in attested_langs
                        if lang.subgroup == p_subgroup
                    ]
                if not descendants:
                    descendants = [
                        lang for lang in attested_langs
                        if (
                            clean_p in (lang.subgroup or "").lower() or
                            clean_p in (lang.family or "").lower()
                        )
                    ]
                if not descendants:
                    descendants = attested_langs
                    
                families = [lang.family for lang in descendants if lang.family]
                if families:
                    from collections import Counter
                    proto_lang.family = Counter(families).most_common(1)[0][0]

        family_groups: dict[str, list[str]] = {}
        for gc, lang in dataset.languages.items():
            family = lang.family or "unknown"
            family_groups.setdefault(family, []).append(gc)

        # Helper to find a node in the tree by label
        def find_node(node: TreeNode, label: str) -> TreeNode | None:
            if node.label == label:
                return node
            for child in node.children:
                res = find_node(child, label)
                if res:
                    return res
            return None

        for family, glottocodes in family_groups.items():
            # Attested languages in the family
            attested_gcs = [
                gc for gc in glottocodes
                if gc in dataset.languages and not dataset.languages[gc].is_proto
            ]
            if len(attested_gcs) < 2:
                continue  # Need at least 2 attested languages

            # Try to get the family Glottocode for tree retrieval.
            family_gc = self._resolve_family_glottocode(attested_gcs)
            if family_gc is None:
                logger.warning(
                    "Could not resolve family Glottocode for '%s' in "
                    "dataset '%s' — skipping.",
                    family,
                    dataset.dataset_name,
                )
                continue

            # Load the full tree first
            try:
                newick_str = self.glottolog.get_family_tree(family_gc)
                full_tree = parse_newick(newick_str)
            except Exception:
                logger.exception(
                    "Failed to get Newick tree for family '%s'.", family_gc
                )
                continue

            if not full_tree:
                continue

            # Build full tree leaf node map
            full_leaf_nodes = {leaf.label: leaf for leaf in full_tree.get_leaves() if leaf.label}

            # Proto-languages in this family
            family_proto_gcs = [
                gc for gc in glottocodes
                if gc in dataset.languages and dataset.languages[gc].is_proto
            ]

            # Map proto-languages to Glottocodes/nodes in the full tree
            mapped_proto_gcs: dict[str, str] = {}

            # 1. Resolve name-based proto glottocodes
            for proto_gc in family_proto_gcs:
                proto_lang = dataset.languages[proto_gc]
                # Check if it has a valid glottocode in Glottolog
                if proto_gc != proto_lang.name and self.glottolog.api.languoid(proto_gc):
                    mapped_proto_gcs[proto_gc] = proto_gc
                    continue
                # Try resolving by name
                resolved_gc = self.proto_name_map.get(proto_lang.name.lower().strip())
                if not resolved_gc:
                    clean_name = clean_proto_name(proto_lang.name)
                    resolved_gc = self.proto_name_map.get(clean_name)
                if resolved_gc:
                    mapped_proto_gcs[proto_gc] = resolved_gc

            # 2. For unresolved ones, map to MRCA of their attested descendants in the unpruned tree
            unresolved_proto_gcs = [gc for gc in family_proto_gcs if gc not in mapped_proto_gcs]
            unresolved_proto_gcs_sorted = sorted(
                unresolved_proto_gcs,
                key=lambda gc: len(dataset.languages[gc].name),
                reverse=True
            )

            for proto_gc in unresolved_proto_gcs_sorted:
                proto_lang = dataset.languages[proto_gc]
                clean_p = clean_proto_name(proto_lang.name)
                p_subgroup = (proto_lang.subgroup or "").strip()

                # Find descendants in dataset
                descendants = []
                if p_subgroup:
                    descendants = [
                        gc for gc in attested_gcs
                        if gc in full_leaf_nodes and dataset.languages[gc].subgroup == p_subgroup
                    ]
                if not descendants:
                    descendants = [
                        gc for gc in attested_gcs
                        if gc in full_leaf_nodes and (
                            clean_p in (dataset.languages[gc].subgroup or "").lower() or
                            clean_p in (dataset.languages[gc].family or "").lower()
                        )
                    ]
                if not descendants:
                    descendants = [gc for gc in attested_gcs if gc in full_leaf_nodes]

                if len(descendants) >= 2:
                    mrca = full_leaf_nodes[descendants[0]]
                    for gc in descendants[1:]:
                        mrca_next = find_mrca(mrca, full_leaf_nodes[gc])
                        if mrca_next:
                            mrca = mrca_next
                    mapped_proto_gcs[proto_gc] = mrca.label

            # The set of glottocodes to preserve in the pruned tree
            keep_gcs = set(attested_gcs) | set(mapped_proto_gcs.values())

            # Prune the tree keeping all of these
            tree = prune_tree(full_tree, keep_gcs)
            if tree is None:
                logger.warning(
                    "Could not get tree for family '%s' (%s) — skipping.",
                    family,
                    family_gc,
                )
                continue

            # Rename target node labels in the pruned tree to the dataset's proto IDs
            for proto_gc, resolved_gc in mapped_proto_gcs.items():
                node = find_node(tree, resolved_gc)
                if node:
                    node.label = proto_gc

            # Resolve polytomies into binary trees.
            binary_trees = resolve_all_polytomies(
                tree,
                max_resolutions_per_node=self.config.max_polytomy_resolutions,
                max_total_trees=1000,
                rng=self._rng,
            )

            # Shuffle binary trees to ensure diversity when we hit the
            # early-exit threshold or the reservoir cap.
            self._rng.shuffle(binary_trees)

            # Early-exit: stop iterating binary trees when consecutive
            # trees yield no new unique triplets (diminishing returns).
            stale_tree_count = 0
            stale_tree_limit = max(10, len(binary_trees) // 10)

            for btree in binary_trees:
                # Dispatch to the task-specific generator.
                if self.config.task == "reconstruction":
                    generator = generate_reconstruction_examples(
                        dataset, btree, self.config, self._rng
                    )
                else:
                    generator = generate_cognate_reflex_examples(
                        dataset, btree, self.config, self._rng
                    )

                found_new_in_tree = False
                for ex in generator:
                    # Deduplicate examples across different binary
                    # resolutions using compact int hashes instead of
                    # full signature tuples.
                    input_gcs = tuple(sorted(i.glottocode for i in ex.inputs))
                    target_gc = ex.target.glottocode
                    cog_ids = tuple(sorted(ex.metadata.cognateset_ids))
                    sig_hash = hash((input_gcs, target_gc, cog_ids))

                    if sig_hash not in seen_hashes:
                        seen_hashes.add(sig_hash)
                        count += 1
                        found_new_in_tree = True

                        if max_triplets is None or len(dataset_examples) < max_triplets:
                            dataset_examples.append(ex)
                        else:
                            j = self._rng.randrange(count)
                            if j < max_triplets:
                                dataset_examples[j] = ex

                if found_new_in_tree:
                    stale_tree_count = 0
                else:
                    stale_tree_count += 1
                    if stale_tree_count >= stale_tree_limit:
                        logger.debug(
                            "Early exit for family '%s' in dataset '%s': "
                            "%d consecutive trees with no new triplets.",
                            family,
                            dataset.dataset_name,
                            stale_tree_count,
                        )
                        break

        if max_triplets is not None and len(dataset_examples) > 0:
            # Although the reservoir sample is a random subset, the first
            # max_triplets items (if count >= max_triplets) might be
            # biased in order.  Shuffle to ensure random order.
            self._rng.shuffle(dataset_examples)

        yield from dataset_examples

    def _resolve_family_glottocode(
        self, glottocodes: list[str]
    ) -> str | None:
        """Find the top-level family Glottocode for a group of languages."""
        for gc in glottocodes:
            family_gc = self.glottolog.get_family_for_language(gc)
            if family_gc is not None:
                return family_gc
        return None

    def _get_pruned_tree(
        self, family_gc: str, keep_glottocodes: set[str]
    ) -> TreeNode | None:
        """Get a Glottolog family tree pruned to the desired languages."""
        try:
            newick_str = self.glottolog.get_family_tree(family_gc)
        except Exception:
            logger.exception(
                "Failed to get Newick tree for family '%s'.", family_gc
            )
            return None

        if not newick_str:
            return None

        try:
            full_tree = parse_newick(newick_str)
        except Exception:
            logger.exception(
                "Failed to parse Newick tree for family '%s'.", family_gc
            )
            return None

        pruned = prune_tree(full_tree, keep_glottocodes)
        return pruned

