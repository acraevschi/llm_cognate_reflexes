"""General-purpose utility functions for the cognate_reflexes package."""

from __future__ import annotations

import random


def normalize_segments(segments: list[str]) -> list[str]:
    """Strip whitespace from each segment, drop empty tokens, and lowercase.

    Args:
        segments: Raw IPA segment tokens (e.g. from a CLDF ``Segments``
            column that has already been split on spaces).

    Returns:
        Cleaned list of non-empty, lowercased segment strings.
    """
    return [s.strip().lower() for s in segments if s.strip()]


def segments_to_string(segments: list[str]) -> str:
    """Join IPA segments into a single space-separated string.

    Args:
        segments: List of IPA segment tokens.

    Returns:
        Space-separated concatenation (e.g. ``"b a n a n a"``).
    """
    return " ".join(segments)


def string_to_segments(s: str) -> list[str]:
    """Split a space-separated IPA string back into segment tokens.

    Args:
        s: Space-separated IPA string.

    Returns:
        List of individual segment tokens.
    """
    return s.split()


def set_random_seed(seed: int | None) -> None:
    """Seed all relevant random-number generators for reproducibility.

    Currently seeds:
    * :mod:`random`

    If *seed* is ``None`` the call is a no-op (randomness is left
    unseeded / OS-entropy-seeded).

    Args:
        seed: Integer seed, or ``None`` to skip seeding.
    """
    if seed is None:
        return
    random.seed(seed)
