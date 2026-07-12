"""Shared strict-model configuration and token conventions."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
MORPHOLOGICAL_BOUNDARY_TOKENS: frozenset[str] = frozenset({"+", "-"})


class WorkbenchModel(BaseModel):
    """Immutable, strict, and closed base class for serialized workbench data."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
    )
