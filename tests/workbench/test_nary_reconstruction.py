from __future__ import annotations

import math

from cognate_reconstruction.ingestion import ingest_payload
from cognate_reconstruction.rules import parse_rule
from cognate_reconstruction.schemas.ingestion import WorkbenchPayload
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm
from cognate_reconstruction.schemas.rules import AnchorPolicy, ReconstructionRule
from cognate_reconstruction.traversal import RuleBasedReconstructor, TreeTraverser
from cognate_reconstruction.traversal.beam import make_leaf_beam


def lexicon(variety_id: str, *segments: tuple[str, ...]) -> LanguageLexicon:
    return LanguageLexicon(
        variety_id=variety_id,
        name=variety_id,
        forms=tuple(
            LexicalForm(
                form_id=f"{variety_id}:water:{index}",
                variety_id=variety_id,
                concept_id="water",
                segments=value,
            )
            for index, value in enumerate(segments)
        ),
    )


def test_native_polytomy_is_one_nary_reconstruction_step() -> None:
    dataset = ingest_payload(
        WorkbenchPayload(
            lexicons=(lexicon("A", ("p",)), lexicon("B", ("p",)), lexicon("C", ("p",))),
            newick="(A,B,C)PROTO;",
        )
    )
    snapshot = TreeTraverser().traverse(dataset)
    assert len(snapshot.steps) == 1
    assert snapshot.steps[0].child_node_ids == ("A", "B", "C")
    assert snapshot.steps[0].output_beam.source_child_ids == ("A", "B", "C")


def test_tree_normalization_prunes_missing_data_and_collapses_unary_nodes() -> None:
    dataset = ingest_payload(
        WorkbenchPayload(
            lexicons=(lexicon("A", ("p",)), lexicon("B", ("p",)), lexicon("C", ("p",))),
            newick="((A,X)UNSUPPORTED,(Y,Z)EMPTY,B,C)ROOT;",
        )
    )
    assert dataset.tree.newick == "(A,B,C)ROOT;"
    assert dataset.tree.leaf_variety_ids == ("A", "B", "C")


def test_tree_normalization_quotes_dataset_scoped_variety_ids() -> None:
    dataset = ingest_payload(
        WorkbenchPayload(
            lexicons=(lexicon("ds:A", ("p",)), lexicon("ds:B", ("p",))),
            newick="('ds:A','ds:B')ROOT;",
        )
    )
    assert dataset.tree.newick == "('ds:A','ds:B')ROOT;"
    assert TreeTraverser().traverse(dataset).root_node_id == "ROOT"


def test_rule_scope_can_target_one_child_in_polytomy() -> None:
    children = tuple(
        make_leaf_beam(item, beam_width=2)
        for item in (
            lexicon("A", ("f",)),
            lexicon("B", ("p",)),
            lexicon("C", ("p",)),
        )
    )
    rule = ReconstructionRule(
        rule=parse_rule("f > p", rule_id="restore-p"),
        source_child_ids=("A",),
        confidence=0.9,
    )
    step = RuleBasedReconstructor(beam_width=2).reconstruct(
        "PROTO", children, rules=(rule,)
    )
    candidate = step.output_beam.distributions[0].candidates[0]
    assert candidate.segments == ("p",)
    assert math.isclose(candidate.log_score, math.log(0.9))


def test_anchor_factor_is_configurable_and_applied_before_pruning() -> None:
    children = tuple(
        make_leaf_beam(item, beam_width=2)
        for item in (
            lexicon("A", ("p",), ("b",)),
            lexicon("B", ("p",), ("b",)),
        )
    )
    anchor = LexicalForm(
        form_id="anchor:water",
        variety_id="PROTO",
        concept_id="water",
        segments=("f",),
    )
    rule = ReconstructionRule(
        rule=parse_rule("p > f", rule_id="frication"),
        source_child_ids=("A", "B"),
        confidence=0.5,
    )
    neutral = RuleBasedReconstructor(
        beam_width=1, anchor_match_factor=1.0
    ).reconstruct("PROTO", children, rules=(rule,), anchors=(anchor,))
    boosted = RuleBasedReconstructor(
        beam_width=1,
        anchor_policy=AnchorPolicy.SCORED,
        anchor_match_factor=100.0,
    ).reconstruct("PROTO", children, rules=(rule,), anchors=(anchor,))
    assert neutral.output_beam.distributions[0].candidates[0].segments == ("b",)
    assert boosted.output_beam.distributions[0].candidates[0].segments == ("f",)
    assert any(
        result.matched_anchor_ids == ("anchor:water",)
        for report in boosted.rule_reports
        for result in report.results
    )


def test_anchors_are_advisory_by_default() -> None:
    children = tuple(
        make_leaf_beam(item, beam_width=2)
        for item in (
            lexicon("A", ("p",), ("b",)),
            lexicon("B", ("p",), ("b",)),
        )
    )
    anchor = LexicalForm(
        form_id="anchor:water",
        variety_id="PROTO",
        concept_id="water",
        segments=("f",),
    )
    rule = ReconstructionRule(
        rule=parse_rule("p > f"),
        source_child_ids=("A", "B"),
        confidence=0.5,
    )
    step = RuleBasedReconstructor(
        beam_width=1,
        anchor_match_factor=100.0,
    ).reconstruct("PROTO", children, rules=(rule,), anchors=(anchor,))
    assert step.output_beam.distributions[0].candidates[0].segments == ("b",)
    assert any(report.anchors_matched for report in step.rule_reports)


def test_incremental_combination_never_exceeds_beam_width() -> None:
    children = tuple(
        make_leaf_beam(lexicon(child, ("p",), ("b",), ("k",)), beam_width=2)
        for child in ("A", "B", "C", "D")
    )
    step = RuleBasedReconstructor(beam_width=2).reconstruct("PROTO", children)
    assert len(step.output_beam.distributions[0].candidates) <= 2


def test_traversal_exposes_only_observed_and_completed_tree_evidence() -> None:
    class CapturingReconstructor:
        def __init__(self) -> None:
            self.base = RuleBasedReconstructor()
            self.contexts = []

        def reconstruct(self, *args: object, **kwargs: object):
            self.contexts.append(kwargs["evidence_context"])
            return self.base.reconstruct(*args, **kwargs)

    dataset = ingest_payload(
        WorkbenchPayload(
            lexicons=tuple(
                lexicon(node_id, ("p",)) for node_id in ("A", "B", "C", "D")
            ),
            newick="((A,B)X,(C,D)Y)ROOT;",
        )
    )
    capturing = CapturingReconstructor()
    TreeTraverser(reconstructor=capturing).traverse(dataset)
    first, second, root = capturing.contexts
    assert {item.node_id for item in first.available_nodes} == {"A", "B", "C", "D"}
    assert "Y" not in {item.node_id for item in second.available_nodes}
    assert "X" in {item.node_id for item in second.available_nodes}
    root_relations = {item.node_id: item.relation.value for item in root.available_nodes}
    assert root_relations["X"] == "active_child"
    assert root_relations["Y"] == "active_child"
