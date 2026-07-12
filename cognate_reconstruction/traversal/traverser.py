"""Bottom-up state manager for normalized native n-ary Newick trees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cognate_reconstruction.schemas.ingestion import IngestedDataset
from cognate_reconstruction.schemas.lexicon import LexicalForm
from cognate_reconstruction.schemas.rules import (
    AnomalyReport,
    ParsedSoundRule,
    ReconstructionRule,
)
from cognate_reconstruction.schemas.traversal import TraversalSnapshot
from cognate_reconstruction.schemas.traversal import (
    EvidenceKind,
    EvidenceRelation,
    NodeEvidence,
    NodeReconstructionContext,
)
from cognate_reconstruction.traversal.beam import beam_to_lexicon, make_leaf_beam
from cognate_reconstruction.traversal.protocol import NodeReconstructor
from cognate_reconstruction.traversal.reconstructor import RuleBasedReconstructor
from cognate_reflexes.tree.newick_utils import TreeNode, parse_newick, postorder_groups


def _assign_node_ids(root: TreeNode) -> dict[int, str]:
    assigned: dict[int, str] = {}
    used: set[str] = set()

    def visit(node: TreeNode, path: str) -> None:
        node_id = node.label or f"internal:{path}"
        if node_id in used:
            raise ValueError(f"tree node identifier {node_id!r} is not unique")
        assigned[id(node)] = node_id
        used.add(node_id)
        for index, child in enumerate(node.children):
            visit(child, f"{path}.{index}")

    visit(root, "root")
    return assigned


class TreeTraverser:
    def __init__(
        self,
        *,
        beam_width: int = 5,
        reconstructor: NodeReconstructor | None = None,
    ) -> None:
        self.beam_width = beam_width
        self.reconstructor = reconstructor or RuleBasedReconstructor(beam_width=beam_width)

    def traverse(
        self,
        dataset: IngestedDataset,
        *,
        rules_by_node: Mapping[
            str, Sequence[ReconstructionRule | ParsedSoundRule]
        ] | None = None,
        anomalies_by_node: Mapping[str, Sequence[AnomalyReport]] | None = None,
        anchors_by_node: Mapping[str, Sequence[LexicalForm]] | None = None,
    ) -> TraversalSnapshot:
        root = parse_newick(dataset.tree.newick)
        node_ids = _assign_node_ids(root)
        lexicons = {lexicon.variety_id: lexicon for lexicon in dataset.lexicons}
        beams = {}
        observed_evidence: dict[str, tuple[LanguageLexicon, tuple[str, ...]]] = {}
        for leaf in root.get_leaves():
            if leaf.label is None or leaf.label not in lexicons:
                raise ValueError(f"no lexicon for tree leaf {leaf.label!r}")
            beams[id(leaf)] = make_leaf_beam(
                lexicons[leaf.label], beam_width=self.beam_width
            )
            observed_evidence[leaf.label] = (lexicons[leaf.label], (leaf.label,))

        steps = []
        completed = []
        node_rules = rules_by_node or {}
        node_anomalies = anomalies_by_node or {}
        node_anchors = anchors_by_node or {}
        reconstructed_evidence: dict[str, tuple[LanguageLexicon, tuple[str, ...]]] = {}
        for children, parent in postorder_groups(root):
            parent_id = node_ids[id(parent)]
            active_child_ids = tuple(node_ids[id(child)] for child in children)
            parent_leaf_ids = tuple(sorted(parent.get_leaf_labels()))
            available_nodes = []
            for kind, evidence_items in (
                (EvidenceKind.OBSERVED, observed_evidence),
                (EvidenceKind.RECONSTRUCTED, reconstructed_evidence),
            ):
                for node_id, (lexicon, descendant_ids) in sorted(evidence_items.items()):
                    if node_id in active_child_ids:
                        relation = EvidenceRelation.ACTIVE_CHILD
                    elif set(descendant_ids) <= set(parent_leaf_ids):
                        relation = EvidenceRelation.DESCENDANT
                    else:
                        relation = EvidenceRelation.OUTGROUP
                    available_nodes.append(
                        NodeEvidence(
                            node_id=node_id,
                            kind=kind,
                            relation=relation,
                            lexicon=lexicon,
                            descendant_leaf_ids=descendant_ids,
                        )
                    )
            evidence_context = NodeReconstructionContext(
                parent_node_id=parent_id,
                active_child_ids=active_child_ids,
                available_nodes=tuple(available_nodes),
                concepts=dataset.concepts,
            )
            step = self.reconstructor.reconstruct(
                parent_id,
                tuple(beams[id(child)] for child in children),
                rules=node_rules.get(parent_id, ()),
                anomalies=node_anomalies.get(parent_id, ()),
                anchors=node_anchors.get(parent_id, ()),
                evidence_context=evidence_context,
            )
            beams[id(parent)] = step.output_beam
            reconstructed_evidence[parent_id] = (
                beam_to_lexicon(step.output_beam),
                parent_leaf_ids,
            )
            steps.append(step)
            completed.append(parent_id)
        return TraversalSnapshot(
            root_node_id=node_ids[id(root)],
            completed_node_ids=tuple(completed),
            node_beams=tuple(beams[key] for key in sorted(beams, key=lambda key: node_ids[key])),
            steps=tuple(steps),
        )
