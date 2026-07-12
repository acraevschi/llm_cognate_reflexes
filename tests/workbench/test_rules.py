from __future__ import annotations

from cognate_reconstruction.rules import RuleEngine, parse_rule
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import ApplicationStatus


def form(form_id: str, segments: tuple[str, ...]) -> LexicalForm:
    return LexicalForm(
        form_id=form_id,
        variety_id="child",
        concept_id="concept",
        segments=segments,
    )


def test_final_rule_diff_and_candidate_provenance() -> None:
    rule = parse_rule("p > f / _#", rule_id="final-frication")
    report = RuleEngine().apply_rule(
        rule,
        (form("matches", ("a", "p")), form("wrong-context", ("p", "a"))),
        source_candidate_ids={"matches": "candidate-7"},
    )
    assert report.words_applied == 1
    assert report.results[0].output_segments == ("a", "f")
    assert report.results[0].source_candidate_id == "candidate-7"
    assert report.results[1].status is ApplicationStatus.CONTEXT_MISMATCH
    assert report.exceptions == (report.results[1],)


def test_morphological_boundary_is_not_transparent() -> None:
    rule = parse_rule("k > tʃ / _i")
    report = RuleEngine().apply_rule(rule, (form("f", ("k", "+", "i")),))
    assert report.results[0].status is ApplicationStatus.CONTEXT_MISMATCH


def test_morphological_boundary_can_be_explicit_context() -> None:
    rule = parse_rule("k > tʃ / _ +")
    report = RuleEngine().apply_rule(rule, (form("f", ("k", "+", "i")),))
    assert report.results[0].output_segments == ("tʃ", "+", "i")


def test_anchor_mismatch_is_mechanically_applied() -> None:
    rule = parse_rule("p > f")
    report = RuleEngine().apply_rule(
        rule,
        (form("anchor", ("p",)),),
        anchor_expected={"anchor": ("v",)},
    )
    assert report.results[0].status is ApplicationStatus.ANCHOR_MISMATCH
    assert report.words_applied == 1
