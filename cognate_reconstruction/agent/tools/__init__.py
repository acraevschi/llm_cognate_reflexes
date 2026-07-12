"""Default deterministic tool registry."""

from cognate_reconstruction.agent.schemas import (
    CommitReconstructionArgs,
    GetAlignmentsArgs,
    ListAvailableNodesArgs,
    ListConceptsArgs,
    SearchFormsArgs,
    SegmentMorphemesArgs,
    TestSoundLawArgs,
)
from cognate_reconstruction.agent.tools.commit_reconstruction import commit_reconstruction
from cognate_reconstruction.agent.tools.get_alignments import get_alignments
from cognate_reconstruction.agent.tools.evidence import (
    list_available_nodes,
    list_concepts,
    search_forms,
)
from cognate_reconstruction.agent.tools.registry import ToolRegistry, ToolSpec
from cognate_reconstruction.agent.tools.segment_morphemes import segment_morphemes
from cognate_reconstruction.agent.tools.test_sound_law import test_sound_law


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="get_alignments",
            description=(
                "Align forms from any two or more available nodes and return an "
                "n-way MSA plus derived pairwise correspondences."
            ),
            args_model=GetAlignmentsArgs,
            handler=get_alignments,
        )
    )
    registry.register(
        ToolSpec(
            name="list_concepts",
            description="List searchable concept metadata and form counts.",
            args_model=ListConceptsArgs,
            handler=list_concepts,
        )
    )
    registry.register(
        ToolSpec(
            name="search_forms",
            description=(
                "Search active or available-tree forms by semantics, segments, "
                "position, node, or cognate set."
            ),
            args_model=SearchFormsArgs,
            handler=search_forms,
        )
    )
    registry.register(
        ToolSpec(
            name="list_available_nodes",
            description="List observed and already reconstructed evidence nodes.",
            args_model=ListAvailableNodesArgs,
            handler=list_available_nodes,
        )
    )
    registry.register(
        ToolSpec(
            name="test_sound_law",
            description="Parse and apply one child-to-parent DSL rule, returning exact diffs.",
            args_model=TestSoundLawArgs,
            handler=test_sound_law,
        )
    )
    registry.register(
        ToolSpec(
            name="segment_morphemes",
            description="Create a temporary boundary-only segmentation overlay.",
            args_model=SegmentMorphemesArgs,
            handler=segment_morphemes,
        )
    )
    registry.register(
        ToolSpec(
            name="commit_reconstruction",
            description="Commit the ordered validated rule cascade and explicit anomalies.",
            args_model=CommitReconstructionArgs,
            handler=commit_reconstruction,
        )
    )
    return registry


__all__ = ["ToolRegistry", "ToolSpec", "default_tool_registry"]
