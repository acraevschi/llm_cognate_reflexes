"""Bottom-up state manager for binary Newick trees."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from cognate_reconstruction.schemas.ingestion import IngestedDataset
from cognate_reconstruction.schemas.rules import AnomalyReport, ParsedSoundRule
from cognate_reconstruction.schemas.traversal import TraversalSnapshot
from cognate_reconstruction.traversal.beam import make_leaf_beam
from cognate_reconstruction.traversal.protocol import NodeReconstructor
from cognate_reconstruction.traversal.reconstructor import RuleBasedReconstructor
from cognate_reflexes.tree.newick_utils import TreeNode, parse_newick, postorder_triplets


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
        rules_by_node: Mapping[str, Sequence[ParsedSoundRule]] | None = None,
        anomalies_by_node: Mapping[str, Sequence[AnomalyReport]] | None = None,
    ) -> TraversalSnapshot:
        root = parse_newick(dataset.tree.newick)
        node_ids = _assign_node_ids(root)
        lexicons = {lexicon.variety_id: lexicon for lexicon in dataset.lexicons}
        beams = {}
        for leaf in root.get_leaves():
            if leaf.label is None or leaf.label not in lexicons:
                raise ValueError(f"no lexicon for tree leaf {leaf.label!r}")
            beams[id(leaf)] = make_leaf_beam(
                lexicons[leaf.label], beam_width=self.beam_width
            )

        steps = []
        completed = []
        node_rules = rules_by_node or {}
        node_anomalies = anomalies_by_node or {}
        for left, right, parent in postorder_triplets(root):
            parent_id = node_ids[id(parent)]
            step = self.reconstructor.reconstruct(
                parent_id,
                beams[id(left)],
                beams[id(right)],
                rules=node_rules.get(parent_id, ()),
                anomalies=node_anomalies.get(parent_id, ()),
            )
            beams[id(parent)] = step.output_beam
            steps.append(step)
            completed.append(parent_id)
        return TraversalSnapshot(
            root_node_id=node_ids[id(root)],
            completed_node_ids=tuple(completed),
            node_beams=tuple(beams[key] for key in sorted(beams, key=lambda key: node_ids[key])),
            steps=tuple(steps),
        )
