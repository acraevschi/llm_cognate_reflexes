"""JSONL serialisation and deserialisation for TrainingExample objects.

Writes each :class:`TrainingExample` as a single JSON line containing:
* ``input_text`` / ``target_text`` — formatted strings (via
  :class:`TextFormatter`).
* ``metadata`` — full provenance information.
* ``raw`` — structured form data for downstream reconstruction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from cognate_reflexes.formatting.formatter import TextFormatter
from cognate_reflexes.examples.models import (
    Form,
    LanguageData,
    TrainingExample,
    ExampleMetadata,
)

logger = logging.getLogger(__name__)


class ExampleSerializer:
    """Serialize and deserialize TrainingExample objects to/from JSONL.

    Args:
        formatter: A :class:`TextFormatter` used to produce
            ``input_text`` / ``target_text`` fields.  Defaults to
            a freshly constructed :class:`TextFormatter`.
    """

    def __init__(self, formatter: TextFormatter | None = None) -> None:
        self.formatter = formatter or TextFormatter()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialize_example(self, example: TrainingExample) -> dict:
        """Convert a :class:`TrainingExample` into a JSON-serialisable dict.

        The dict contains:
        * ``input_text`` / ``target_text``: formatted strings.
        * ``task``: task label.
        * ``masked_indices``: which target forms are hidden.
        * ``metadata``: provenance.
        * ``raw``: structured form data (inputs, target).
        """
        input_text, target_text = self.formatter.format_example(example)

        return {
            "input_text": input_text,
            "target_text": target_text,
            "task": example.task,
            "masked_indices": example.masked_indices,
            "metadata": self._metadata_to_dict(example.metadata),
            "raw": {
                "inputs": [self._language_to_dict(inp) for inp in example.inputs],
                "target": self._language_to_dict(example.target),
            },
        }

    def write_jsonl(
        self, examples: Iterator[TrainingExample], path: str | Path
    ) -> int:
        """Write examples to a JSONL file.

        Args:
            examples: Iterator of :class:`TrainingExample` objects.
            path: Output file path.

        Returns:
            Number of examples written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(path, "w", encoding="utf-8") as fh:
            for example in examples:
                record = self.serialize_example(example)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                if count % 1000 == 0:
                    logger.info("Written %d examples…", count)
        logger.info("Finished writing %d examples to '%s'.", count, path)
        return count

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    def read_jsonl(self, path: str | Path) -> Iterator[dict]:
        """Read examples from a JSONL file.

        Yields raw dicts (not :class:`TrainingExample` objects) for maximum
        flexibility — callers can pick the fields they need.

        Args:
            path: Path to the JSONL file.

        Yields:
            Dicts as written by :meth:`write_jsonl`.
        """
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    # ------------------------------------------------------------------
    # Dict helpers
    # ------------------------------------------------------------------

    def _form_to_dict(self, form: Form) -> dict:
        """Convert a :class:`Form` to a serialisable dict."""
        return {
            "form_id": form.form_id,
            "language": form.language,
            "language_name": form.language_name,
            "segments": list(form.segments),
            "concept": form.concept,
            "concepticon_id": form.concepticon_id,
            "cognateset_id": form.cognateset_id,
            "dataset": form.dataset,
        }

    def _language_to_dict(self, lang: LanguageData) -> dict:
        """Convert a :class:`LanguageData` to a serialisable dict."""
        return {
            "glottocode": lang.glottocode,
            "name": lang.name,
            "forms": [self._form_to_dict(f) for f in lang.forms],
            "latitude": lang.latitude,
            "longitude": lang.longitude,
            "family": lang.family,
            "is_proto": lang.is_proto,
        }

    def _metadata_to_dict(self, meta: ExampleMetadata) -> dict:
        """Convert :class:`ExampleMetadata` to a serialisable dict."""
        return {
            "source_dataset": meta.source_dataset,
            "language_family": meta.language_family,
            "tree_depth": meta.tree_depth,
            "branch_lengths": meta.branch_lengths,
            "num_cognate_sets": meta.num_cognate_sets,
            "glottocodes": list(meta.glottocodes),
            "coordinates": {
                k: list(v) if v else None
                for k, v in meta.coordinates.items()
            },
            "concept_ids": meta.concept_ids,
            "cognateset_ids": meta.cognateset_ids,
        }

