"""Typed wrapper around LingPy pairwise and multiple sequence alignment."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, product
from typing import Literal

from cognate_reconstruction.schemas.alignment import (
    AlignmentMember,
    AlignmentResult,
    CorrespondenceMap,
    CorrespondenceObservation,
    CorrespondenceSummary,
    MultipleAlignmentMap,
)
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm


def _context(sequence: tuple[str | None, ...], index: int) -> tuple[str | None, str | None]:
    before = sequence[index - 1] if index > 0 else None
    after = sequence[index + 1] if index + 1 < len(sequence) else None
    return before, after


def _pairwise_map(
    left_id: str,
    right_id: str,
    alignments: Sequence[AlignmentResult],
) -> CorrespondenceMap:
    relevant = tuple(
        alignment
        for alignment in alignments
        if {left_id, right_id}
        <= {member.variety_id for member in alignment.members if not member.is_anchor}
    )
    observations: dict[
        tuple[str | None, str | None], list[CorrespondenceObservation]
    ] = defaultdict(list)
    for alignment in relevant:
        left_members = [m for m in alignment.members if m.variety_id == left_id]
        right_members = [m for m in alignment.members if m.variety_id == right_id]
        anchor_present = any(m.is_anchor for m in alignment.members)
        for left_member, right_member in product(left_members, right_members):
            for index, (left_segment, right_segment) in enumerate(
                zip(left_member.aligned_segments, right_member.aligned_segments, strict=True)
            ):
                observations[(left_segment, right_segment)].append(
                    CorrespondenceObservation(
                        alignment_id=alignment.alignment_id,
                        column_index=index,
                        left_segment=left_segment,
                        right_segment=right_segment,
                        left_context=_context(left_member.aligned_segments, index),
                        right_context=_context(right_member.aligned_segments, index),
                        anchor_supported=anchor_present,
                    )
                )
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
        left_variety_id=left_id,
        right_variety_id=right_id,
        alignments=relevant,
        correspondences=summaries,
    )


class LingPyAligner:
    """Align cognate-aware concept groups across two or more lexicons."""

    def __init__(
        self,
        *,
        method: Literal["sca"] = "sca",
        mode: Literal["global", "local", "overlap", "dialign"] = "global",
    ) -> None:
        if method != "sca":
            raise ValueError("LingPyAligner currently supports only the SCA model")
        if mode not in {"global", "local", "overlap", "dialign"}:
            raise ValueError(f"unsupported LingPy alignment mode {mode!r}")
        self.method = method
        self.mode = mode

    def align(
        self,
        left: LanguageLexicon,
        right: LanguageLexicon,
        anchors: tuple[LexicalForm, ...] = (),
    ) -> CorrespondenceMap:
        """Compatibility pairwise view derived from the same n-way engine."""
        result = self.align_multiple((left, right), anchors)
        return result.pairwise_correspondences[0]

    def align_multiple(
        self,
        lexicons: Sequence[LanguageLexicon],
        anchors: tuple[LexicalForm, ...] = (),
        *,
        respect_cognate_sets: bool = True,
    ) -> MultipleAlignmentMap:
        from lingpy import Multiple  # type: ignore[import-untyped]

        selected = tuple(lexicons)
        variety_ids = tuple(lexicon.variety_id for lexicon in selected)
        if len(variety_ids) < 2 or len(set(variety_ids)) != len(variety_ids):
            raise ValueError("alignment requires at least two distinct lexicons")

        grouped: dict[
            tuple[str, str | None], list[tuple[LexicalForm, bool]]
        ] = defaultdict(list)
        for lexicon in selected:
            for form in lexicon.forms:
                cognate_id = form.cognate_set_id if respect_cognate_sets else None
                grouped[(form.concept_id, cognate_id)].append((form, False))

        alignments: list[AlignmentResult] = []
        for (concept_id, cognate_set_id), forms_and_flags in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")
        ):
            present = {form.variety_id for form, _ in forms_and_flags}
            if len(present) < 2:
                continue
            compatible_anchors = tuple(
                anchor
                for anchor in anchors
                if anchor.concept_id == concept_id
                and (
                    not respect_cognate_sets
                    or anchor.cognate_set_id is None
                    or cognate_set_id is None
                    or anchor.cognate_set_id == cognate_set_id
                )
            )
            material = forms_and_flags + [(anchor, True) for anchor in compatible_anchors]
            multiple = Multiple(
                [list(form.phonetic_segments) for form, _ in material]
            )
            multiple.prog_align(model="sca", mode=self.mode)
            members = tuple(
                AlignmentMember(
                    form_id=form.form_id,
                    variety_id=form.variety_id,
                    concept_id=concept_id,
                    cognate_set_id=form.cognate_set_id,
                    aligned_segments=tuple(
                        None if token == "-" else str(token) for token in aligned
                    ),
                    is_anchor=is_anchor,
                )
                for (form, is_anchor), aligned in zip(
                    material, multiple.alm_matrix, strict=True
                )
            )
            group_suffix = cognate_set_id or "unassigned"
            alignments.append(
                AlignmentResult(
                    alignment_id=(
                        f"{':'.join(variety_ids)}:{concept_id}:{group_suffix}"
                    ),
                    concept_id=concept_id,
                    cognate_set_id=cognate_set_id,
                    members=members,
                    method=self.method,
                    mode=self.mode,
                )
            )

        pairwise = tuple(
            _pairwise_map(left_id, right_id, alignments)
            for left_id, right_id in combinations(variety_ids, 2)
        )
        return MultipleAlignmentMap(
            variety_ids=variety_ids,
            alignments=tuple(alignments),
            pairwise_correspondences=pairwise,
        )
