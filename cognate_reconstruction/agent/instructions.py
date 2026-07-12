"""Load the versioned behavioral instructions packaged with the agent."""

from __future__ import annotations

from importlib.resources import files


def load_agent_instructions() -> str:
    return files("cognate_reconstruction.agent").joinpath("SKILL.md").read_text(
        encoding="utf-8"
    )
