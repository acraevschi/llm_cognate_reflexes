"""Helper utilities for splitting examples by language family.

The recommended evaluation strategy is to hold out entire language
families for the test set, ensuring the model is evaluated on its
ability to generalise to unseen families.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from cognate_reflexes.examples.models import TrainingExample


def split_by_family(
    examples: Iterable[TrainingExample],
    test_families: list[str],
    val_families: list[str] | None = None,
    val_ratio: float = 0.1,
    seed: int | None = 42,
) -> tuple[list[TrainingExample], list[TrainingExample], list[TrainingExample]]:
    """Split examples into train / val / test by language family.

    Args:
        examples: Iterable of :class:`TrainingExample` objects.
        test_families: Language family names or Glottocodes to hold out
            for the test set.
        val_families: Optional list of families for the validation set.
            When ``None``, *val_ratio* of the remaining (non-test)
            families are randomly sampled.
        val_ratio: Fraction of non-test families to use for validation
            (only used when *val_families* is ``None``).
        seed: Random seed for reproducibility (used when sampling
            validation families).

    Returns:
        ``(train_examples, val_examples, test_examples)``
    """
    rng = random.Random(seed)
    test_set = set(test_families)

    # Group examples by family.
    by_family: dict[str, list[TrainingExample]] = defaultdict(list)
    for example in examples:
        family = example.metadata.language_family
        by_family[family].append(example)

    # If val_families not provided, sample from non-test families.
    if val_families is None:
        non_test_families = [f for f in by_family if f not in test_set]
        num_val = max(1, int(len(non_test_families) * val_ratio))
        num_val = min(num_val, len(non_test_families))
        val_families_sampled = set(rng.sample(non_test_families, num_val))
    else:
        val_families_sampled = set(val_families)

    train: list[TrainingExample] = []
    val: list[TrainingExample] = []
    test: list[TrainingExample] = []

    for family, family_examples in by_family.items():
        if family in test_set:
            test.extend(family_examples)
        elif family in val_families_sampled:
            val.extend(family_examples)
        else:
            train.extend(family_examples)

    return train, val, test


def get_available_families(examples: Iterable[TrainingExample]) -> dict[str, int]:
    """Count examples per language family.

    Useful for deciding which families to hold out for evaluation.

    Args:
        examples: Iterable of :class:`TrainingExample` objects.

    Returns:
        Mapping from family name to count of examples.
    """
    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example.metadata.language_family] += 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

