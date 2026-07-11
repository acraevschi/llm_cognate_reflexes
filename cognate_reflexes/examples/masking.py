"""Random masking utilities for the cognate reflex task.

During cognate-reflex training, some target-language forms are *revealed*
as anchors while the rest are *masked* and must be predicted by the model.
This module provides the selection logic.
"""

from __future__ import annotations

import random


def apply_masking(
    num_forms: int,
    mask_ratio: float = 0.3,
    min_revealed: int = 3,
    rng: random.Random | None = None,
) -> list[int]:
    """Select which target-language form indices to mask.

    Args:
        num_forms: Total number of forms in the target language.
        mask_ratio: Fraction of forms to mask (0.0–1.0).
        min_revealed: Minimum number of forms that must remain visible
            (anchor forms).  If ``num_forms`` is too small this
            constraint takes priority over *mask_ratio*.
        rng: Optional :class:`random.Random` instance for
            reproducibility.  When ``None`` the module-level RNG is
            used.

    Returns:
        Sorted list of indices into the target form list that should
        be masked.  Indices are 0-based.

    Raises:
        ValueError: If *num_forms* < 1 or *mask_ratio* is outside
            [0, 1].
    """
    if num_forms < 1:
        raise ValueError(f"num_forms must be >= 1, got {num_forms}")
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    rng = rng or random.Random()

    # How many forms to mask.
    num_to_mask = int(round(num_forms * mask_ratio))

    # Ensure at least min_revealed forms remain visible.
    max_maskable = max(0, num_forms - min_revealed)
    num_to_mask = min(num_to_mask, max_maskable)

    # Ensure at least 1 form is masked (otherwise there's nothing to predict).
    num_to_mask = max(num_to_mask, min(1, max_maskable))

    all_indices = list(range(num_forms))
    masked = sorted(rng.sample(all_indices, num_to_mask))
    return masked
