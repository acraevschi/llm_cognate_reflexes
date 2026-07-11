"""Generate reconstruction examples with attested historical targets."""

from __future__ import annotations

import itertools
import logging
import random
from typing import Iterator

from cognate_reflexes.config import Config
from cognate_reflexes.data.historical import HistoricalLineageManifest
from cognate_reflexes.data.loader import DatasetForms
from cognate_reflexes.examples.models import ExampleMetadata, TrainingExample
from cognate_reflexes.examples.reconstruction import (
    _build_language_data,
    _shared_cognate_sets_nary,
)

logger = logging.getLogger(__name__)


def _is_eligible_descendant(
    dataset: DatasetForms,
    target_id: str,
    descendant_id: str,
    min_age_gap: float,
) -> bool:
    """Check identity, data availability and any available temporal ordering."""
    if descendant_id == target_id:
        return False
    target = dataset.languages.get(target_id)
    descendant = dataset.languages.get(descendant_id)
    if target is None or descendant is None:
        return False
    if descendant_id not in dataset.forms_by_language:
        return False

    # A manifest supplies the primary lineage evidence.  Dates are an
    # additional guard where both varieties provide comparable estimates.
    if (
        target.date_before_present is not None
        and descendant.date_before_present is not None
        and target.date_before_present <= descendant.date_before_present + min_age_gap
    ):
        return False
    return True


def generate_historical_reconstruction_examples(
    dataset: DatasetForms,
    manifest: HistoricalLineageManifest,
    config: Config,
    rng: random.Random | None = None,
    automatic_lineages: dict[str, dict[str, set[str]]] | None = None,
) -> Iterator[TrainingExample]:
    """Yield full-mask reconstruction examples for validated historical targets.

    Two inputs always come from different first-diverging child branches.
    Consequently a single-chain history (for example, Old Polish → Middle
    Polish → modern Polish) cannot create an example.
    """
    rng = rng or random.Random()

    automatic_lineages = automatic_lineages or {}
    target_ids = sorted(
        set(manifest.targets_for(dataset.dataset_name)) | set(automatic_lineages)
    )
    for target_id in target_ids:
        target = dataset.languages.get(target_id)
        if target is None:
            logger.warning(
                "Historical manifest target '%s' in '%s' is not an available "
                "source variety.",
                target_id,
                dataset.dataset_name,
            )
            continue
        if target_id not in dataset.forms_by_language:
            continue

        # Curated relations take precedence; automatic extraction is used only
        # when no target-specific manual override exists.
        branches = manifest.branches_for(dataset.dataset_name, target_id)
        if not branches:
            branches = automatic_lineages.get(target_id, {})
        eligible_branches = {
            branch_id: sorted(
                descendant_id
                for descendant_id in descendants
                if _is_eligible_descendant(
                    dataset,
                    target_id,
                    descendant_id,
                    config.min_historical_age_gap,
                )
            )
            for branch_id, descendants in branches.items()
        }
        eligible_branches = {
            branch_id: descendants
            for branch_id, descendants in eligible_branches.items()
            if descendants
        }
        if len(eligible_branches) < 2:
            logger.info(
                "Historical target '%s' in '%s' has fewer than two usable "
                "descendant branches — skipping.",
                target_id,
                dataset.dataset_name,
            )
            continue

        candidate_pairs = []
        for left_branch, right_branch in itertools.combinations(
            sorted(eligible_branches), 2
        ):
            candidate_pairs.extend(
                (left_id, right_id, left_branch, right_branch)
                for left_id, right_id in itertools.product(
                    eligible_branches[left_branch],
                    eligible_branches[right_branch],
                )
            )
        rng.shuffle(candidate_pairs)

        # Cap branch-pair expansion before form materialisation.  This keeps
        # large historical targets from dominating a dataset's reservoir.
        for left_id, right_id, left_branch, right_branch in candidate_pairs[:20]:
            input_ids = [left_id, right_id]
            all_ids = input_ids + [target_id]
            shared_cogsets = _shared_cognate_sets_nary(dataset, all_ids)
            if len(shared_cogsets) < config.min_cognates:
                continue
            if len(shared_cogsets) > config.max_cognates:
                shared_cogsets = sorted(
                    rng.sample(shared_cogsets, config.max_cognates)
                )
            rng.shuffle(shared_cogsets)

            inputs = [
                _build_language_data(dataset, variety_id, shared_cogsets, rng)
                for variety_id in input_ids
            ]
            target_data = _build_language_data(
                dataset, target_id, shared_cogsets, rng
            )
            # An explicit manifest relation is authoritative for datasets
            # whose CLDF has no standard ``historical`` field.
            target_data.is_historical = True

            coordinates = {}
            for variety_id in all_ids:
                language = dataset.languages[variety_id]
                coordinates[variety_id] = (
                    (language.latitude, language.longitude)
                    if language.latitude is not None and language.longitude is not None
                    else None
                )

            concept_ids = []
            for cognateset_id in shared_cogsets:
                forms = [
                    dataset.forms_by_language[variety_id].get(cognateset_id, [])
                    for variety_id in all_ids
                ]
                concept_ids.append(
                    next(
                        (
                            form_list[0].concepticon_id
                            for form_list in forms
                            if form_list and form_list[0].concepticon_id
                        ),
                        "",
                    )
                )

            yield TrainingExample(
                task="reconstruction",
                inputs=inputs,
                target=target_data,
                masked_indices=list(range(len(target_data.forms))),
                metadata=ExampleMetadata(
                    source_dataset=dataset.dataset_name,
                    language_family=dataset.family or "unknown",
                    tree_depth=0,
                    branch_lengths=[None, None],
                    num_cognate_sets=len(shared_cogsets),
                    glottocodes=tuple(
                        dataset.languages[variety_id].glottocode
                        for variety_id in all_ids
                    ),
                    variety_ids=tuple(all_ids),
                    coordinates=coordinates,
                    concept_ids=concept_ids,
                    cognateset_ids=shared_cogsets,
                    target_kind="historical",
                    historical_branch_ids=(left_branch, right_branch),
                ),
            )
