"""Initial payload and tree-artifact schemas."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from cognate_reconstruction.schemas.common import NonEmptyStr, WorkbenchModel
from cognate_reconstruction.schemas.lexicon import LanguageLexicon


class TreeOrigin(StrEnum):
    PROVIDED = "provided"
    INDUCED = "induced"


class DistanceMatrix(WorkbenchModel):
    taxa: tuple[NonEmptyStr, ...] = Field(min_length=2)
    values: tuple[tuple[float, ...], ...]
    method: NonEmptyStr

    @model_validator(mode="after")
    def validate_square_symmetric_matrix(self) -> DistanceMatrix:
        size = len(self.taxa)
        if len(set(self.taxa)) != size or len(self.values) != size:
            raise ValueError("distance matrix taxa must be unique and dimensions square")
        for i, row in enumerate(self.values):
            if len(row) != size:
                raise ValueError("distance matrix dimensions must be square")
            if abs(row[i]) > 1e-9:
                raise ValueError("distance matrix diagonal must be zero")
            for j, value in enumerate(row):
                if value < 0 or abs(value - self.values[j][i]) > 1e-9:
                    raise ValueError("distance matrix must be non-negative and symmetric")
        return self


class TreeArtifact(WorkbenchModel):
    newick: NonEmptyStr
    origin: TreeOrigin
    leaf_variety_ids: tuple[NonEmptyStr, ...]
    induction_method: Literal["neighbor", "upgma"] | None = None
    distance_matrix: DistanceMatrix | None = None

    @model_validator(mode="after")
    def validate_origin_metadata(self) -> TreeArtifact:
        if len(set(self.leaf_variety_ids)) != len(self.leaf_variety_ids):
            raise ValueError("tree leaf IDs must be unique")
        if self.origin is TreeOrigin.INDUCED and self.induction_method is None:
            raise ValueError("induced trees require induction_method")
        if self.origin is TreeOrigin.PROVIDED and self.induction_method is not None:
            raise ValueError("provided trees cannot declare induction_method")
        return self


class WorkbenchPayload(WorkbenchModel):
    lexicons: tuple[LanguageLexicon, ...] = Field(min_length=2)
    newick: str | None = None
    tree_method: Literal["neighbor", "upgma"] = "neighbor"
    random_seed: int = 0

    @model_validator(mode="after")
    def validate_varieties(self) -> WorkbenchPayload:
        ids = [lexicon.variety_id for lexicon in self.lexicons]
        if len(ids) != len(set(ids)):
            raise ValueError("lexicon variety IDs must be unique")
        if self.newick is not None and not self.newick.strip():
            raise ValueError("newick must be non-empty when supplied")
        return self


class IngestedDataset(WorkbenchModel):
    lexicons: tuple[LanguageLexicon, ...]
    tree: TreeArtifact
