"""Typed wrapper around LingPy multiple sequence alignment."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Literal

from cognate_reconstruction.schemas.alignment import (
    AlignmentMember,
    AlignmentResult,
    CorrespondenceMap,
    CorrespondenceObservation,
    CorrespondenceSummary,
)
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm


def _context(sequence: tuple[str | None, ...], index: int) -> tuple[str | None, str | None]:
    before = sequence[index - 1] if index > 0 else None
    after = sequence[index + 1] if index + 1 < len(sequence) else None
    return before, after


class LingPyAligner:
    """Align concept groups and expose auditable column correspondences."""

    def __init__(
        self,
        *,
        method: Literal["sca", "lexstat"] = "sca",
        mode: Literal["global", "local", "overlap", "dialign"] = "global",
    ) -> None:
        self.method = method
        self.mode = mode

    def align(
        self,
        left: LanguageLexicon,
        right: LanguageLexicon,
        anchors: tuple[LexicalForm, ...] = (),
    ) -> CorrespondenceMap:
        from lingpy import Multiple  # type: ignore[import-untyped]

        by_concept: dict[str, list[tuple[LexicalForm, bool]]] = defaultdict(list)
        for form in left.forms + right.forms:
            by_concept[form.concept_id].append((form, False))
        for form in anchors:
            by_concept[form.concept_id].append((form, True))

        alignments: list[AlignmentResult] = []
        for concept_id in sorted(by_concept):
            forms_and_flags = by_concept[concept_id]
            varieties = {
                form.variety_id
                for form, is_anchor in forms_and_flags
                if not is_anchor
            }
            if left.variety_id not in varieties or right.variety_id not in varieties:
                continue
            sequences = [list(form.phonetic_segments) for form, _ in forms_and_flags]
            multiple = Multiple(sequences)
            multiple.prog_align(model="sca", mode=self.mode)
            matrix = multiple.alm_matrix
            members: list[AlignmentMember] = []
            for (form, is_anchor), aligned in zip(forms_and_flags, matrix, strict=True):
                members.append(
                    AlignmentMember(
                        form_id=form.form_id,
                        variety_id=form.variety_id,
                        concept_id=concept_id,
                        cognate_set_id=form.cognate_set_id,
                        aligned_segments=tuple(None if token == "-" else str(token) for token in aligned),
                        is_anchor=is_anchor,
                    )
                )
            alignments.append(
                AlignmentResult(
                    alignment_id=f"{left.variety_id}:{right.variety_id}:{concept_id}",
                    concept_id=concept_id,
                    members=tuple(members),
                    method=self.method,
                    mode=self.mode,
                )
            )

        observations: dict[tuple[str | None, str | None], list[CorrespondenceObservation]] = defaultdict(list)
        for alignment in alignments:
            left_members = [m for m in alignment.members if m.variety_id == left.variety_id]
            right_members = [m for m in alignment.members if m.variety_id == right.variety_id]
            anchor_present = any(m.is_anchor for m in alignment.members)
            for left_member, right_member in product(left_members, right_members):
                for index, (left_segment, right_segment) in enumerate(
                    zip(left_member.aligned_segments, right_member.aligned_segments, strict=True)
                ):
                    observation = CorrespondenceObservation(
                        alignment_id=alignment.alignment_id,
                        column_index=index,
                        left_segment=left_segment,
                        right_segment=right_segment,
                        left_context=_context(left_member.aligned_segments, index),
                        right_context=_context(right_member.aligned_segments, index),
                        anchor_supported=anchor_present,
                    )
                    observations[(left_segment, right_segment)].append(observation)

        summaries = tuple(
            CorrespondenceSummary(
                left_segment=pair[0],
                right_segment=pair[1],
                count=len(items),
                anchor_count=sum(item.anchor_supported for item in items),
                observations=tuple(items),
            )
            for pair, items in sorted(
                observations.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
            )
        )
        return CorrespondenceMap(
            left_variety_id=left.variety_id,
            right_variety_id=right.variety_id,
            alignments=tuple(alignments),
            correspondences=summaries,
        )
