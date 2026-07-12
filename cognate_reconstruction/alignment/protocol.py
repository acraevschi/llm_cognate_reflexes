"""Structural interface for replaceable alignment implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from cognate_reconstruction.schemas.alignment import CorrespondenceMap, MultipleAlignmentMap
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm


class AlignmentProvider(Protocol):
    def align(
        self,
        left: LanguageLexicon,
        right: LanguageLexicon,
        anchors: tuple[LexicalForm, ...] = (),
    ) -> CorrespondenceMap: ...

    def align_multiple(
        self,
        lexicons: Sequence[LanguageLexicon],
        anchors: tuple[LexicalForm, ...] = (),
        *,
        respect_cognate_sets: bool = True,
    ) -> MultipleAlignmentMap: ...
