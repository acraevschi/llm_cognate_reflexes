"""Numerically stable beam construction, merging, and pruning."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable

from cognate_reconstruction.schemas.beam import (
    CandidateDerivation,
    ConceptCandidateDistribution,
    NodeBeamState,
    ReconstructionCandidate,
)
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm

RawCandidate = tuple[tuple[str, ...], float, CandidateDerivation]


def _logsumexp(values: Iterable[float]) -> float:
    materialized = tuple(values)
    maximum = max(materialized)
    return maximum + math.log(sum(math.exp(value - maximum) for value in materialized))


def _candidate_id(node_id: str, concept_id: str, segments: tuple[str, ...]) -> str:
    digest = hashlib.sha256("\u241f".join(segments).encode()).hexdigest()[:12]
    return f"{node_id}:{concept_id}:{digest}"


def normalize_and_prune(
    node_id: str,
    concept_id: str,
    raw_candidates: Iterable[RawCandidate],
    *,
    beam_width: int,
) -> ConceptCandidateDistribution:
    """Merge identical strings, retain top N by log mass, and normalize."""
    grouped_scores: dict[tuple[str, ...], list[float]] = defaultdict(list)
    grouped_derivations: dict[tuple[str, ...], list[CandidateDerivation]] = defaultdict(list)
    for segments, log_score, derivation in raw_candidates:
        if not segments or not math.isfinite(log_score):
            continue
        grouped_scores[segments].append(log_score)
        grouped_derivations[segments].append(derivation)
    if not grouped_scores:
        raise ValueError(f"no viable candidates for concept {concept_id!r}")
    merged = sorted(
        ((segments, _logsumexp(scores)) for segments, scores in grouped_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )[:beam_width]
    normalizer = _logsumexp(score for _, score in merged)
    candidates = tuple(
        ReconstructionCandidate(
            candidate_id=_candidate_id(node_id, concept_id, segments),
            segments=segments,
            probability=math.exp(score - normalizer),
            log_score=score,
            derivations=tuple(grouped_derivations[segments]),
        )
        for segments, score in merged
    )
    return ConceptCandidateDistribution(concept_id=concept_id, candidates=candidates)


def make_leaf_beam(lexicon: LanguageLexicon, *, beam_width: int) -> NodeBeamState:
    """Represent observed forms as an initial per-concept distribution."""
    by_concept: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for form in lexicon.forms:
        by_concept[form.concept_id].append(form.segments)
    distributions: list[ConceptCandidateDistribution] = []
    for concept_id, sequences in sorted(by_concept.items()):
        log_prior = -math.log(len(sequences))
        raw = (
            (
                segments,
                log_prior,
                CandidateDerivation(
                    derivation_id=f"observed:{lexicon.variety_id}:{concept_id}:{index}",
                    child_candidate_ids=(),
                    note="observed leaf form",
                ),
            )
            for index, segments in enumerate(sequences)
        )
        distributions.append(
            normalize_and_prune(
                lexicon.variety_id, concept_id, raw, beam_width=beam_width
            )
        )
    return NodeBeamState(
        node_id=lexicon.variety_id,
        distributions=tuple(distributions),
        beam_width=beam_width,
    )


def beam_to_lexicon(beam: NodeBeamState) -> LanguageLexicon:
    """Expose every retained candidate as a read-only node lexicon."""
    return LanguageLexicon(
        variety_id=beam.node_id,
        name=beam.node_id,
        forms=tuple(
            LexicalForm(
                form_id=candidate.candidate_id,
                variety_id=beam.node_id,
                concept_id=distribution.concept_id,
                segments=candidate.segments,
            )
            for distribution in beam.distributions
            for candidate in distribution.candidates
        ),
    )
