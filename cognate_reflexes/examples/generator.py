"""Main ExampleGenerator API — the primary entry point for the package.

Ties together data loading, tree management, and example generation into
a single convenient class that can be used from a fine-tuning script::

    from cognate_reflexes import ExampleGenerator, Config

    gen = ExampleGenerator(config=Config(task="cognate_reflex"))
    for example in gen.generate():
        print(example)
"""

from __future__ import annotations

import hashlib
import logging
import random
import traceback
from typing import Iterator

from cognate_reflexes.config import Config
from cognate_reflexes.data.historical import HistoricalLineageManifest
from cognate_reflexes.data.loader import CLDFLoader, DatasetForms
from cognate_reflexes.data.registry import DatasetRegistry
from cognate_reflexes.data.temporal_trees import (
    TemporalTreeManifest,
    discover_temporal_lineages,
)
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
from cognate_reflexes.examples.historical_reconstruction import (
    generate_historical_reconstruction_examples,
)

logger = logging.getLogger(__name__)


def _stream_dataset_worker(
    dataset_path: str,
    config: Config,
    result_queue: object,
    batch_size: int = 8,
) -> None:
    """Generate one dataset and send bounded example batches to the parent."""
    gen = ExampleGenerator(config)

    # Create a deterministic but unique seed for this dataset to avoid correlated
    # randomness across worker processes.
    path_hash = int(hashlib.md5(str(dataset_path).encode('utf-8')).hexdigest(), 16)
    dataset_seed = ((config.seed or 0) + path_hash) % (2**32)
    gen._rng = random.Random(dataset_seed)
    try:
        ds = gen.loader.load_dataset(dataset_path)
        if ds is not None:
            batch: list[TrainingExample] = []
            for example in gen._generate_for_dataset(ds):
                batch.append(example)
                if len(batch) >= batch_size:
                    result_queue.put(("batch", dataset_path, batch))  # type: ignore[attr-defined]
                    batch = []
            if batch:
                result_queue.put(("batch", dataset_path, batch))  # type: ignore[attr-defined]
    except Exception:
        result_queue.put(("error", dataset_path, traceback.format_exc()))  # type: ignore[attr-defined]
    finally:
        result_queue.put(("done", dataset_path, None))  # type: ignore[attr-defined]


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
        self._historical_manifest: HistoricalLineageManifest | None = None
        self._temporal_tree_manifest: TemporalTreeManifest | None = None

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

    @property
    def historical_manifest(self) -> HistoricalLineageManifest:
        """Load curated historical relations only when reconstruction needs it."""
        if self._historical_manifest is None:
            self._historical_manifest = HistoricalLineageManifest.from_csv(
                self.config.historical_lineages_path
            )
        return self._historical_manifest

    @property
    def temporal_tree_manifest(self) -> TemporalTreeManifest:
        """Load optional paths to authoritative time-aware Newick trees."""
        if self._temporal_tree_manifest is None:
            self._temporal_tree_manifest = TemporalTreeManifest.from_csv(
                self.config.temporal_trees_path
            )
        return self._temporal_tree_manifest

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

        effective_workers = max(1, workers)
        if effective_workers > 4:
            logger.warning(
                "Using %d workers. Each active worker loads a complete CLDF "
                "dataset and tree collection; values above 4 can cause high "
                "memory use.",
                effective_workers,
            )

        if effective_workers <= 1:
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
            # A ProcessPoolExecutor result is materialised before it can be
            # returned.  With thousands of rich examples per dataset that
            # creates several full in-memory copies.  Use a bounded queue of
            # small batches instead, while retaining process isolation.
            import multiprocessing as mp
            from queue import Empty

            paths = [str(ds_info.path) for ds_info in datasets]
            context = mp.get_context("spawn")
            result_queue = context.Queue(maxsize=max(1, effective_workers))
            pending_paths = iter(paths)
            active: dict[str, object] = {}

            def start_next() -> bool:
                try:
                    path = next(pending_paths)
                except StopIteration:
                    return False
                process = context.Process(
                    target=_stream_dataset_worker,
                    args=(path, self.config, result_queue),
                )
                process.start()
                active[path] = process
                return True

            for _ in range(min(effective_workers, len(paths))):
                start_next()

            try:
                while active:
                    try:
                        event, path, payload = result_queue.get(timeout=1)
                    except Empty:
                        # A process can terminate before publishing its final
                        # sentinel (for example after an OOM kill).  Detect it
                        # instead of waiting forever on the queue.
                        for dead_path, process in list(active.items()):
                            if not process.is_alive():
                                process.join()
                                active.pop(dead_path)
                                logger.error(
                                    "Worker for dataset '%s' exited without "
                                    "a completion sentinel.",
                                    dead_path,
                                )
                                start_next()
                        continue
                    if event == "batch":
                        examples = payload
                        logger.debug(
                            "Received %d examples from dataset '%s'.",
                            len(examples),
                            path,
                        )
                        yield from examples
                    elif event == "error":
                        logger.error(
                            "Worker failed processing dataset '%s':\n%s",
                            path,
                            payload,
                        )
                    elif event == "done":
                        process = active.pop(path, None)
                        if process is not None:
                            process.join()
                        start_next()
            finally:
                for process in active.values():
                    if process.is_alive():
                        process.terminate()
                    process.join()
                result_queue.close()

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
        with_historical = self.registry.filter(has_historical_forms=True)

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
            "num_datasets_with_historical": len(with_historical),
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
            manifest_datasets = (
                self.historical_manifest.datasets()
                if self.config.include_historical
                else set()
            )
            temporal_tree_datasets = (
                self.temporal_tree_manifest.datasets()
                if self.config.include_temporal_trees
                else set()
            )
            return [
                info
                for info in self.registry.filter(has_cognates=True)
                if info.has_proto_forms
                or (
                    self.config.include_historical
                    and (
                        info.has_historical_forms or info.name in manifest_datasets
                    )
                )
                or (
                    self.config.include_temporal_trees
                    and (
                        info.has_source_tree
                        or info.name in temporal_tree_datasets
                    )
                )
            ]
        else:
            return self.registry.filter(has_cognates=True)

    def _generate_for_dataset(
        self, dataset: DatasetForms
    ) -> Iterator[TrainingExample]:
        """Generate training examples for one dataset with bounded memory.

        Examples are yielded immediately after deduplication.  We deliberately
        avoid a per-dataset reservoir of complete examples: each object
        contains raw forms and can be large enough to make a 5,000-item
        reservoir dominate worker memory.
        """
        seen_hashes: set[int] = set()
        emitted = 0
        max_triplets = self.config.max_triplets_per_dataset

        def is_new_example(example: TrainingExample) -> bool:
            """Deduplicate without retaining the full example object."""
            input_ids = tuple(sorted(item.identifier for item in example.inputs))
            target_id = example.target.identifier
            cognateset_ids = tuple(sorted(example.metadata.cognateset_ids))
            signature = hash((input_ids, target_id, cognateset_ids))
            if signature in seen_hashes:
                return False
            seen_hashes.add(signature)
            return True

        # Clean proto name helper
        import re
        def clean_proto_name(name: str) -> str:
            name = name.lower()
            name = re.sub(r'^[Pp]roto[- ]', '', name)
            name = name.replace('*', '').strip()
            return name

        # Group languages by family.
        # First, resolve missing family names for proto-languages based on their descendants.
        for proto_id in dataset.proto_languages:
            proto_lang = dataset.languages[proto_id]
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
        if not (
            self.config.task == "reconstruction" and not dataset.proto_languages
        ):
            for variety_id, lang in dataset.languages.items():
                family = lang.family or "unknown"
                family_groups.setdefault(family, []).append(variety_id)

        # Helper to find a node in the tree by label
        def find_node(node: TreeNode, label: str) -> TreeNode | None:
            if node.label == label:
                return node
            for child in node.children:
                res = find_node(child, label)
                if res:
                    return res
            return None

        for family, variety_ids in family_groups.items():
            # Attested languages in the family
            attested_ids = [
                variety_id for variety_id in variety_ids
                if (
                    variety_id in dataset.languages
                    and not dataset.languages[variety_id].is_proto
                )
            ]
            if len(attested_ids) < 2:
                continue  # Need at least 2 attested languages

            # Try to get the family Glottocode for tree retrieval.
            family_gc = self._resolve_family_glottocode(dataset, attested_ids)
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
            family_proto_ids = [
                variety_id for variety_id in variety_ids
                if (
                    variety_id in dataset.languages
                    and dataset.languages[variety_id].is_proto
                )
            ]

            # Map proto-languages to Glottocodes/nodes in the full tree
            mapped_proto_ids: dict[str, str] = {}

            # 1. Resolve name-based proto glottocodes
            for proto_id in family_proto_ids:
                proto_lang = dataset.languages[proto_id]
                # Check if it has a valid glottocode in Glottolog
                if (
                    proto_lang.tree_glottocode
                    and self.glottolog.api.languoid(proto_lang.tree_glottocode)
                ):
                    mapped_proto_ids[proto_id] = proto_lang.tree_glottocode
                    continue
                # Try resolving by name
                resolved_gc = self.proto_name_map.get(proto_lang.name.lower().strip())
                if not resolved_gc:
                    clean_name = clean_proto_name(proto_lang.name)
                    resolved_gc = self.proto_name_map.get(clean_name)
                if resolved_gc:
                    mapped_proto_ids[proto_id] = resolved_gc

            # 2. For unresolved ones, map to MRCA of their attested descendants in the unpruned tree
            unresolved_proto_ids = [
                variety_id
                for variety_id in family_proto_ids
                if variety_id not in mapped_proto_ids
            ]
            unresolved_proto_ids_sorted = sorted(
                unresolved_proto_ids,
                key=lambda variety_id: len(dataset.languages[variety_id].name),
                reverse=True
            )

            for proto_id in unresolved_proto_ids_sorted:
                proto_lang = dataset.languages[proto_id]
                clean_p = clean_proto_name(proto_lang.name)
                p_subgroup = (proto_lang.subgroup or "").strip()

                # Find descendants in dataset
                descendants = []
                if p_subgroup:
                    descendants = [
                        variety_id for variety_id in attested_ids
                        if (
                            dataset.languages[variety_id].tree_glottocode
                            in full_leaf_nodes
                            and dataset.languages[variety_id].subgroup == p_subgroup
                        )
                    ]
                if not descendants:
                    descendants = [
                        variety_id for variety_id in attested_ids
                        if (
                            dataset.languages[variety_id].tree_glottocode
                            in full_leaf_nodes
                            and (
                                clean_p
                                in (dataset.languages[variety_id].subgroup or "").lower()
                                or clean_p
                                in (dataset.languages[variety_id].family or "").lower()
                            )
                        )
                    ]
                if not descendants:
                    descendants = [
                        variety_id
                        for variety_id in attested_ids
                        if dataset.languages[variety_id].tree_glottocode
                        in full_leaf_nodes
                    ]

                if len(descendants) >= 2:
                    first_gc = dataset.languages[descendants[0]].tree_glottocode
                    assert first_gc is not None
                    mrca = full_leaf_nodes[first_gc]
                    for variety_id in descendants[1:]:
                        descendant_gc = dataset.languages[variety_id].tree_glottocode
                        assert descendant_gc is not None
                        mrca_next = find_mrca(mrca, full_leaf_nodes[descendant_gc])
                        if mrca_next:
                            mrca = mrca_next
                    if mrca.label:
                        mapped_proto_ids[proto_id] = mrca.label

            # The set of glottocodes to preserve in the pruned tree
            keep_gcs = {
                dataset.languages[variety_id].tree_glottocode
                for variety_id in attested_ids
                if dataset.languages[variety_id].tree_glottocode
            } | set(mapped_proto_ids.values())

            # Prune the tree keeping all of these
            tree = prune_tree(full_tree, keep_gcs)
            if tree is None:
                logger.warning(
                    "Could not get tree for family '%s' (%s) — skipping.",
                    family,
                    family_gc,
                )
                continue

            # One Glottocode can refer to several source varieties.  A
            # Glottolog leaf can represent only an unambiguous variety; skip
            # collisions here rather than merging their forms.  Historical
            # examples use the lineage manifest and remain available.
            tree_to_varieties: dict[str, list[str]] = {}
            for variety_id in attested_ids:
                tree_gc = dataset.languages[variety_id].tree_glottocode
                if tree_gc:
                    tree_to_varieties.setdefault(tree_gc, []).append(variety_id)
            for tree_gc, mapped_variety_ids in tree_to_varieties.items():
                if len(mapped_variety_ids) != 1:
                    logger.info(
                        "Dataset '%s' has %d varieties for Glottocode '%s'; "
                        "excluding that leaf from Glottolog-derived examples.",
                        dataset.dataset_name,
                        len(mapped_variety_ids),
                        tree_gc,
                    )
                    continue
                node = find_node(tree, tree_gc)
                if node:
                    node.label = mapped_variety_ids[0]

            # Rename target nodes to their dataset-scoped variety IDs.
            for proto_id, resolved_gc in mapped_proto_ids.items():
                node = find_node(tree, resolved_gc)
                if node:
                    node.label = proto_id

            # Resolve polytomies into binary trees.
            binary_trees = resolve_all_polytomies(
                tree,
                max_resolutions_per_node=self.config.max_polytomy_resolutions,
                max_total_trees=self.config.max_total_binary_trees,
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
                    if is_new_example(ex):
                        found_new_in_tree = True
                        emitted += 1
                        yield ex
                        if max_triplets is not None and emitted >= max_triplets:
                            return

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

        if self.config.task == "reconstruction" and self.config.include_historical:
            automatic_lineages = (
                discover_temporal_lineages(dataset, self.temporal_tree_manifest)
                if self.config.include_temporal_trees
                else {}
            )
            for example in generate_historical_reconstruction_examples(
                dataset,
                self.historical_manifest,
                self.config,
                self._rng,
                automatic_lineages=automatic_lineages,
            ):
                if is_new_example(example):
                    emitted += 1
                    yield example
                    if max_triplets is not None and emitted >= max_triplets:
                        return

    def _resolve_family_glottocode(
        self, dataset: DatasetForms, variety_ids: list[str]
    ) -> str | None:
        """Find the top-level family Glottocode for a group of languages."""
        for variety_id in variety_ids:
            language = dataset.languages[variety_id]
            tree_glottocode = language.tree_glottocode
            if not tree_glottocode:
                continue
            family_gc = self.glottolog.get_family_for_language(tree_glottocode)
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
