"""Read-only concept, form, and tree-evidence discovery tools."""

from __future__ import annotations

from collections import defaultdict

from cognate_reconstruction.agent.context import AgentContext
from cognate_reconstruction.agent.schemas import (
    AvailableNodeSummary,
    ConceptListing,
    EvidenceScope,
    FormSearchHit,
    ListAvailableNodesArgs,
    ListAvailableNodesResult,
    ListConceptsArgs,
    ListConceptsResult,
    SearchFormsArgs,
    SearchFormsResult,
    SegmentPosition,
)
from cognate_reconstruction.schemas.common import WorkbenchModel
from cognate_reconstruction.schemas.lexicon import ConceptMetadata
from cognate_reconstruction.schemas.traversal import (
    EvidenceKind,
    EvidenceRelation,
    NodeEvidence,
)


def _concept_catalog(context: AgentContext) -> dict[str, ConceptMetadata]:
    return {concept.concept_id: concept for concept in context.concepts}


def _fallback_concept(concept_id: str) -> ConceptMetadata:
    return ConceptMetadata(concept_id=concept_id, gloss=concept_id)


def _concept_text(concept: ConceptMetadata) -> str:
    return " ".join(
        item
        for item in (
            concept.concept_id,
            concept.gloss,
            concept.concepticon_id,
            concept.semantic_field,
            *concept.aliases,
        )
        if item is not None
    ).casefold()


def _selected_evidence(
    context: AgentContext,
    scope: EvidenceScope,
    node_ids: tuple[str, ...],
) -> tuple[NodeEvidence, ...]:
    selected_ids = set(node_ids)
    evidence_by_id = {item.node_id: item for item in context.evidence}
    if scope is EvidenceScope.AVAILABLE_TREE:
        items = tuple(
            item.model_copy(update={"lexicon": context.lexicon(item.node_id)})
            if item.node_id in context.child_ids
            else item
            for item in context.evidence
        )
    else:
        items = tuple(
            NodeEvidence(
                node_id=lexicon.variety_id,
                kind=(
                    evidence_by_id[lexicon.variety_id].kind
                    if lexicon.variety_id in evidence_by_id
                    else EvidenceKind.RECONSTRUCTED
                ),
                relation=EvidenceRelation.ACTIVE_CHILD,
                lexicon=lexicon,
                descendant_leaf_ids=(
                    evidence_by_id[lexicon.variety_id].descendant_leaf_ids
                    if lexicon.variety_id in evidence_by_id
                    else ()
                ),
            )
            for lexicon in context.child_lexicons
        )
    if selected_ids:
        unknown = selected_ids - {item.node_id for item in items}
        if unknown:
            raise ValueError(f"nodes are unavailable in the selected scope: {sorted(unknown)}")
        items = tuple(item for item in items if item.node_id in selected_ids)
    return items


def list_concepts(
    raw_arguments: WorkbenchModel,
    context: AgentContext,
    call_id: str,  # noqa: ARG001
) -> ListConceptsResult:
    arguments = ListConceptsArgs.model_validate(raw_arguments)
    evidence = _selected_evidence(context, arguments.scope, arguments.node_ids)
    catalog = _concept_catalog(context)
    occurrences: dict[str, list[str]] = defaultdict(list)
    for item in evidence:
        for form in item.lexicon.forms:
            occurrences[form.concept_id].append(item.node_id)
    query = arguments.query.casefold() if arguments.query else None
    listings = []
    for concept_id, node_ids in sorted(occurrences.items()):
        concept = catalog.get(concept_id, _fallback_concept(concept_id))
        if query and query not in _concept_text(concept):
            continue
        listings.append(
            ConceptListing(
                concept=concept,
                form_count=len(node_ids),
                node_ids=tuple(sorted(set(node_ids))),
            )
        )
    page = listings[arguments.offset : arguments.offset + arguments.limit]
    next_offset = arguments.offset + len(page)
    return ListConceptsResult(
        concepts=tuple(page),
        next_offset=next_offset if next_offset < len(listings) else None,
    )


def _segments_match(
    sequence: tuple[str, ...],
    pattern: tuple[str, ...],
    position: SegmentPosition,
) -> bool:
    if not pattern:
        return True
    width = len(pattern)
    if position is SegmentPosition.INITIAL:
        return sequence[:width] == pattern
    if position is SegmentPosition.FINAL:
        return sequence[-width:] == pattern
    if position is SegmentPosition.EXACT:
        return sequence == pattern
    return any(
        sequence[index : index + width] == pattern
        for index in range(len(sequence) - width + 1)
    )


def search_forms(
    raw_arguments: WorkbenchModel,
    context: AgentContext,
    call_id: str,  # noqa: ARG001
) -> SearchFormsResult:
    arguments = SearchFormsArgs.model_validate(raw_arguments)
    evidence = _selected_evidence(context, arguments.scope, arguments.node_ids)
    catalog = _concept_catalog(context)
    selected_concepts = set(arguments.concept_ids)
    selected_cognates = set(arguments.cognate_set_ids)
    query = arguments.concept_query.casefold() if arguments.concept_query else None
    hits = []
    for item in evidence:
        for form in item.lexicon.forms:
            concept = catalog.get(form.concept_id, _fallback_concept(form.concept_id))
            if selected_concepts and form.concept_id not in selected_concepts:
                continue
            if query and query not in _concept_text(concept):
                continue
            if selected_cognates and form.cognate_set_id not in selected_cognates:
                continue
            sequence = form.segments if arguments.include_boundaries else form.phonetic_segments
            if not _segments_match(sequence, arguments.segment_pattern, arguments.position):
                continue
            hits.append(
                FormSearchHit(
                    node_id=item.node_id,
                    evidence_kind=item.kind,
                    relation=item.relation,
                    concept=concept,
                    form=form,
                )
            )
    hits.sort(key=lambda hit: (hit.concept.concept_id, hit.node_id, hit.form.form_id))
    page = hits[arguments.offset : arguments.offset + arguments.limit]
    next_offset = arguments.offset + len(page)
    return SearchFormsResult(
        hits=tuple(page),
        next_offset=next_offset if next_offset < len(hits) else None,
    )


def list_available_nodes(
    raw_arguments: WorkbenchModel,
    context: AgentContext,
    call_id: str,  # noqa: ARG001
) -> ListAvailableNodesResult:
    arguments = ListAvailableNodesArgs.model_validate(raw_arguments)
    kinds = set(arguments.kinds)
    relations = set(arguments.relations)
    nodes = tuple(
        AvailableNodeSummary(
            node_id=item.node_id,
            kind=item.kind,
            relation=item.relation,
            descendant_leaf_ids=item.descendant_leaf_ids,
            form_count=len(item.lexicon.forms),
            concept_count=len({form.concept_id for form in item.lexicon.forms}),
        )
        for item in context.evidence
        if (not kinds or item.kind in kinds)
        and (not relations or item.relation in relations)
    )
    return ListAvailableNodesResult(nodes=nodes)
