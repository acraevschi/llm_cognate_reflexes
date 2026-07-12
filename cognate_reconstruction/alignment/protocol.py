"""Structural interface for replaceable alignment implementations."""

from __future__ import annotations

from typing import Protocol

from cognate_reconstruction.schemas.alignment import CorrespondenceMap
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm


class AlignmentProvider(Protocol):
    def align(
        self,
        left: LanguageLexicon,
        right: LanguageLexicon,
        anchors: tuple[LexicalForm, ...] = (),
    ) -> CorrespondenceMap: ...
