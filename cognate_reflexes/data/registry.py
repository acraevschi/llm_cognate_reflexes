"""Dataset discovery and lightweight metadata cataloguing.

:class:`DatasetRegistry` provides a fast scanning pass over a directory of
Lexibank CLDF datasets.  Instead of loading every form (which can be slow
for hundreds of datasets), it inspects only the CLDF metadata JSON and the
``languages.csv`` to determine whether a dataset has cognate information,
proto-language forms, etc.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ======================================================================
# Dataset descriptor
# ======================================================================

@dataclass
class DatasetInfo:
    """Lightweight metadata for a single CLDF dataset.

    Populated during :meth:`DatasetRegistry.scan` without loading the
    full form table.

    Attributes:
        name: Dataset directory name (repository name).
        path: Absolute path to the dataset directory.
        has_cognates: ``True`` if the dataset contains cognate-set
            information (inline or via CognateTable).
        has_proto_forms: ``True`` if at least one language name in
            ``languages.csv`` looks like a proto-language.
        num_languages: Number of rows in ``languages.csv``.
        num_concepts: Number of rows in the parameter (concept) table,
            if present (``0`` otherwise).
        num_forms: Number of rows in the form table, if countable cheaply
            (``0`` otherwise — this is a fast scan, not a full load).
        families: Distinct language-family names found in ``languages.csv``.
    """

    name: str
    path: Path
    has_cognates: bool = False
    has_proto_forms: bool = False
    num_languages: int = 0
    num_concepts: int = 0
    num_forms: int = 0
    families: list[str] = field(default_factory=list)


# ======================================================================
# Registry
# ======================================================================

class DatasetRegistry:
    """Discover and catalogue CLDF datasets in a directory tree.

    Args:
        data_dir: Root directory containing one subdirectory per dataset.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._datasets: dict[str, DatasetInfo] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> None:
        """Scan *data_dir* and populate the internal catalogue.

        For each subdirectory, the method checks whether a valid CLDF
        metadata JSON exists and inspects it to determine cognate-set
        availability.  It also reads ``languages.csv`` for proto-language
        names and family information.

        Calling :meth:`scan` again re-scans from scratch.
        """
        self._datasets.clear()

        if not self.data_dir.exists():
            logger.warning("Data directory '%s' does not exist.", self.data_dir)
            return

        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            info = self._inspect(subdir)
            if info is not None:
                self._datasets[info.name] = info

        logger.info("Registry scan complete: %d dataset(s).", len(self._datasets))

    def filter(
        self,
        *,
        has_cognates: bool | None = None,
        has_proto_forms: bool | None = None,
        family: str | None = None,
        min_languages: int | None = None,
    ) -> list[DatasetInfo]:
        """Return datasets matching the given criteria.

        All filters are optional; unset filters are ignored.

        Args:
            has_cognates: If ``True``, include only datasets with cognate
                information.
            has_proto_forms: If ``True``, include only datasets with at
                least one proto-language.
            family: If set, include only datasets whose *families* list
                contains this value (case-insensitive substring match).
            min_languages: If set, include only datasets with at least
                this many languages.

        Returns:
            A list of matching :class:`DatasetInfo` objects, sorted by name.
        """
        results: list[DatasetInfo] = []

        for info in self._datasets.values():
            if has_cognates is not None and info.has_cognates != has_cognates:
                continue
            if (
                has_proto_forms is not None
                and info.has_proto_forms != has_proto_forms
            ):
                continue
            if family is not None:
                family_lower = family.lower()
                if not any(
                    family_lower in f.lower() for f in info.families
                ):
                    continue
            if (
                min_languages is not None
                and info.num_languages < min_languages
            ):
                continue
            results.append(info)

        return sorted(results, key=lambda d: d.name)

    def list_all(self) -> list[DatasetInfo]:
        """Return all discovered datasets, sorted by name."""
        return sorted(self._datasets.values(), key=lambda d: d.name)

    def get(self, name: str) -> DatasetInfo | None:
        """Look up a dataset by name.

        Args:
            name: Dataset directory name.

        Returns:
            The :class:`DatasetInfo` if found, otherwise ``None``.
        """
        return self._datasets.get(name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @classmethod
    def _inspect(cls, dataset_dir: Path) -> DatasetInfo | None:
        """Perform a lightweight inspection of a single dataset directory.

        This examines the CLDF metadata JSON schema and reads
        ``languages.csv`` / ``concepts.csv`` directly (as plain CSV) to
        avoid the overhead of constructing full pycldf objects.

        Returns:
            A populated :class:`DatasetInfo`, or ``None`` if the directory
            does not contain a recognisable CLDF dataset.
        """
        metadata_path = cls._find_metadata(dataset_dir)
        if metadata_path is None:
            return None

        try:
            with open(metadata_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            logger.debug("Could not parse metadata: %s", metadata_path)
            return None

        cldf_dir = metadata_path.parent

        # ---- Determine cognate availability ---------------------------
        has_cognate_table = False
        has_inline_cognateset = False
        forms_csv_name: str | None = None
        languages_csv_name: str | None = None
        parameters_csv_name: str | None = None

        tables = meta.get("tables", [])
        for table in tables:
            table_url = table.get("url", "")
            dc_type = table.get("dc:conformsTo", "")

            # Identify table kind
            if "CognateTable" in dc_type or "cognates" in table_url.lower():
                has_cognate_table = True

            if "FormTable" in dc_type or table_url.lower() in (
                "forms.csv",
                "values.csv",
            ):
                forms_csv_name = table_url
                # Check for inline Cognateset_ID column
                for col in table.get("tableSchema", {}).get("columns", []):
                    col_name = col.get("name", "")
                    prop_url = col.get("propertyUrl", "")
                    if (
                        col_name == "Cognateset_ID"
                        or "cognatesetReference" in prop_url
                    ):
                        has_inline_cognateset = True
                        break

            if "ParameterTable" in dc_type or table_url.lower() in (
                "parameters.csv",
                "concepts.csv",
            ):
                parameters_csv_name = table_url

            if "LanguageTable" in dc_type or table_url.lower() in (
                "languages.csv",
            ):
                languages_csv_name = table_url

        has_cognates = has_cognate_table or has_inline_cognateset

        # ---- Read languages.csv ---------------------------------------
        num_languages = 0
        families: list[str] = []
        has_proto_forms = False

        lang_csv = cls._resolve_csv(cldf_dir, languages_csv_name, "languages")
        if lang_csv is not None:
            try:
                num_languages, families, has_proto_forms = cls._scan_languages(
                    lang_csv,
                )
            except Exception:
                logger.debug(
                    "Error reading languages file: %s", lang_csv, exc_info=True,
                )

        # ---- Count concepts -------------------------------------------
        num_concepts = 0
        params_csv = cls._resolve_csv(
            cldf_dir, parameters_csv_name, "parameters", "concepts",
        )
        if params_csv is not None:
            num_concepts = cls._count_csv_rows(params_csv)

        # ---- Count forms (cheap line count) ---------------------------
        num_forms = 0
        forms_csv = cls._resolve_csv(cldf_dir, forms_csv_name, "forms")
        if forms_csv is not None:
            num_forms = cls._count_csv_rows(forms_csv)

        return DatasetInfo(
            name=dataset_dir.name,
            path=dataset_dir,
            has_cognates=has_cognates,
            has_proto_forms=has_proto_forms,
            num_languages=num_languages,
            num_concepts=num_concepts,
            num_forms=num_forms,
            families=families,
        )

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_metadata(dataset_dir: Path) -> Path | None:
        """Locate CLDF metadata JSON, matching the loader's logic."""
        cldf_dir = dataset_dir / "cldf"
        if cldf_dir.is_dir():
            for candidate in sorted(cldf_dir.glob("*-metadata.json")):
                return candidate
            simple = cldf_dir / "cldf-metadata.json"
            if simple.exists():
                return simple

        for candidate in sorted(dataset_dir.glob("*-metadata.json")):
            return candidate

        return None

    @staticmethod
    def _resolve_csv(
        cldf_dir: Path,
        declared_name: str | None,
        *fallback_stems: str,
    ) -> Path | None:
        """Resolve a CSV path, trying the declared filename first, then
        common fallback names."""
        if declared_name:
            candidate = cldf_dir / declared_name
            if candidate.exists():
                return candidate

        for stem in fallback_stems:
            candidate = cldf_dir / f"{stem}.csv"
            if candidate.exists():
                return candidate

        return None

    @staticmethod
    def _scan_languages(
        csv_path: Path,
    ) -> tuple[int, list[str], bool]:
        """Read a languages CSV and extract count, families, proto flag.

        Returns:
            ``(num_languages, unique_families, has_proto_forms)``
        """
        families_seen: set[str] = set()
        has_proto = False
        count = 0

        with open(csv_path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                count += 1
                name = (row.get("Name") or "").strip()
                family = (row.get("Family") or "").strip()
                if family:
                    families_seen.add(family)
                lower_name = name.lower()
                if lower_name.startswith("proto-") or lower_name.startswith(
                    "proto "
                ):
                    has_proto = True

        return count, sorted(families_seen), has_proto

    @staticmethod
    def _count_csv_rows(csv_path: Path) -> int:
        """Count data rows in a CSV file (excluding header)."""
        try:
            with open(csv_path, encoding="utf-8", newline="") as fh:
                # Use a raw line count minus 1 for the header.
                # This is much faster than parsing every row.
                total = sum(1 for _ in fh)
                return max(total - 1, 0)
        except OSError:
            return 0
