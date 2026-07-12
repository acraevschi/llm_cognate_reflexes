from __future__ import annotations

from cognate_reconstruction.ingestion import ingest_payload
from cognate_reconstruction.rules import parse_rule
from cognate_reconstruction.schemas.ingestion import WorkbenchPayload
from cognate_reconstruction.schemas.lexicon import LanguageLexicon, LexicalForm
from cognate_reconstruction.schemas.rules import AnomalyReport, AnomalyType
from cognate_reconstruction.traversal import TreeTraverser


def lexicon(variety_id: str, segments: tuple[str, ...]) -> LanguageLexicon:
    return LanguageLexicon(
        variety_id=variety_id,
        name=variety_id,
        forms=(
            LexicalForm(
                form_id=f"{variety_id}:water",
                variety_id=variety_id,
                concept_id="water",
                segments=segments,
            ),
        ),
    )


def test_bottom_up_beam_and_step_anomalies() -> None:
    dataset = ingest_payload(
        WorkbenchPayload(
            lexicons=(lexicon("A", ("p",)), lexicon("B", ("p",))),
            newick="(A,B)PROTO;",
        )
    )
    anomaly = AnomalyReport(
        concept_id="water",
        anomaly_type=AnomalyType.UNKNOWN_IRREGULARITY,
        explanation="manual linguistic review requested",
    )
    snapshot = TreeTraverser(beam_width=2).traverse(
        dataset,
        rules_by_node={"PROTO": (parse_rule("p > f"),)},
        anomalies_by_node={"PROTO": (anomaly,)},
    )
    assert snapshot.root_node_id == "PROTO"
    step = snapshot.steps[0]
    assert step.output_beam.distributions[0].candidates[0].segments == ("f",)
    assert step.anomaly_reports == (anomaly,)
    assert all(
        result.source_candidate_id is not None
        for report in step.rule_reports
        for result in report.results
    )
