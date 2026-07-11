"""Interface to pyglottolog for Glottolog classification tree access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class GlottologTree:
    """Access Glottolog classification trees via pyglottolog.

    Provides a thin wrapper around the pyglottolog ``Glottolog`` API,
    exposing only the subset needed for cognate-reflex analysis:
    Newick family trees (with Glottocode labels), languoid lookup,
    and family-membership queries.

    Example::

        gt = GlottologTree("/path/to/glottolog")
        nwk = gt.get_family_tree("indo1319")   # Indo-European
        name = gt.get_languoid_name("stan1293") # "Standard German"
    """

    def __init__(self, glottolog_dir: str | Path) -> None:
        """Initialise with the path to a cloned Glottolog repository.

        Args:
            glottolog_dir: Filesystem path to the root of the cloned
                `glottolog/glottolog <https://github.com/glottolog/glottolog>`_
                repository.  The directory must contain the ``languoids/``
                sub-tree.

        Raises:
            ImportError: If ``pyglottolog`` is not installed.
            ValueError: If *glottolog_dir* does not point to a valid
                Glottolog repository.
        """
        from pyglottolog import Glottolog  # type: ignore[import-untyped]

        self.api: Glottolog = Glottolog(str(glottolog_dir))
        self._family_cache: dict[str, str] = {}  # glottocode -> newick

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_family_tree(self, family_glottocode: str) -> str:
        """Return the Newick tree string for a language family.

        Node labels in the returned Newick string are **Glottocodes**
        (produced with ``template='{l.id}'``).

        Args:
            family_glottocode: Glottocode of the top-level family node
                (e.g. ``"indo1319"`` for Indo-European).

        Returns:
            A Newick-format string whose leaf/node labels are Glottocodes.
        """
        if family_glottocode not in self._family_cache:
            newick_str: str = self.api.newick_tree(
                start=family_glottocode,
                template="{l.id}",
            )
            self._family_cache[family_glottocode] = newick_str
            logger.debug(
                "Cached Newick tree for family %s (%d chars)",
                family_glottocode,
                len(newick_str),
            )
        return self._family_cache[family_glottocode]

    def get_family_for_language(self, glottocode: str) -> str | None:
        """Return the top-level family Glottocode for a language.

        Walks up the parent chain until it reaches a node whose
        ``parent`` is ``None`` (the family root).

        Args:
            glottocode: Glottocode of the language to look up.

        Returns:
            The Glottocode of the top-level family, or ``None`` if the
            languoid could not be found.
        """
        lang: Any = self.api.languoid(glottocode)
        if lang is None:
            logger.warning("Languoid not found: %s", glottocode)
            return None

        current = lang
        while current.parent:
            current = current.parent
        return current.id  # type: ignore[no-any-return]

    def get_languoid_name(self, glottocode: str) -> str | None:
        """Return the human-readable name for a Glottocode.

        Args:
            glottocode: Glottocode to look up.

        Returns:
            The languoid name (e.g. ``"Standard German"``), or ``None``
            if the glottocode is not found.
        """
        lang: Any = self.api.languoid(glottocode)
        if lang is None:
            return None
        return lang.name  # type: ignore[no-any-return]

    def list_families(self) -> list[tuple[str, str]]:
        """List all top-level language families.

        Returns:
            A list of ``(glottocode, name)`` pairs for every languoid
            whose ``level`` is ``"family"`` and whose ``parent`` is
            ``None``.
        """
        families: list[tuple[str, str]] = []
        for languoid in self.api.languoids():
            if languoid.level.name == "family" and languoid.parent is None:
                families.append((languoid.id, languoid.name))
        return families
