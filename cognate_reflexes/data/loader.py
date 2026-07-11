"""Load CLDF datasets into domain models for downstream triplet generation.

This module provides :class:`CLDFLoader`, which reads Lexibank-style CLDF
datasets using `pycldf <https://github.com/cldf/pycldf>`_, resolves cognate
set membership (from either inline ``Cognateset_ID`` in ``FormTable`` or a
separate ``CognateTable``), and organises the results into
:class:`DatasetForms` structures that the triplet generator can consume
directly.

Each dataset is treated **independently** — there is no cross-dataset
merging of cognate sets or language inventories.
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pycldf import Dataset

from cognate_reflexes.examples.models import Form, LanguageData

logger = logging.getLogger(__name__)


def _apply_dataset_overrides(
    ds_name: str,
    language_data: dict[str, LanguageData],
    forms_by_language: dict[str, dict[str, list[Form]]],
    proto_languages: set[str],
) -> None:
    # 1. Glottocode overrides (e.g. for tlopo, tuled)
    LANGUAGE_OVERRIDES = {
        "tlopo": {
            "pan": "aust1307",
            "poc": "ocea1241",
        },
        "tuled": {
            "Tenharim": "tenh1241",
            "Wirafed": "wira1264",
            "OldGuarani": "oldg1234",
            "Kampe": "camp1260",
            "Ramarama": "itog1239",
            "Apapokuva": "apap1239",
            "Piripkura": "piri1253",
            "Kawahiva": "kawa1283",
            "Karipuna": "kari1312",
            "MaweNatterer": "sate1243",
            "MundurukuNatterer": "mund1330",
            "ApiakaNatterer": "apia1248",
            "Arawine": "araw1282",
        }
    }

    if ds_name in LANGUAGE_OVERRIDES:
        for local_id, target_gc in LANGUAGE_OVERRIDES[ds_name].items():
            for gc_key, lang in list(language_data.items()):
                if gc_key == local_id or lang.name == local_id:
                    lang.glottocode = target_gc
                    if gc_key in forms_by_language:
                        forms_dict = forms_by_language[gc_key]
                        for forms in forms_dict.values():
                            for f in forms:
                                f.language = target_gc
                        forms_by_language[target_gc] = forms_by_language.pop(gc_key)
                    
                    language_data[target_gc] = language_data.pop(gc_key)
                    if gc_key in proto_languages:
                        proto_languages.remove(gc_key)
                        proto_languages.add(target_gc)
                    break

    # 2. Cognate set linking overrides (e.g. for sidwellvietic)
    if ds_name == "sidwellvietic":
        param_cog_langs = {}
        for gc, forms_dict in forms_by_language.items():
            for cid, forms in forms_dict.items():
                if forms and forms[0].concepticon_id:
                    param = forms[0].concepticon_id
                    param_cog_langs.setdefault(param, {}).setdefault(cid, []).extend(forms)

        for param, cog_map in param_cog_langs.items():
            proto_cids = []
            attested_cids_with_counts = []
            for cid, forms in cog_map.items():
                is_proto = any(f.language == "viet1250" for f in forms)
                if is_proto:
                    proto_cids.append(cid)
                else:
                    attested_cids_with_counts.append((cid, len(forms)))
                    
            if proto_cids and attested_cids_with_counts:
                major_cid = max(attested_cids_with_counts, key=lambda x: x[1])[0]
                for proto_cid in proto_cids:
                    if proto_cid in forms_by_language.get("viet1250", {}):
                        proto_forms = forms_by_language["viet1250"].pop(proto_cid)
                        for f in proto_forms:
                            f.cognateset_id = major_cid
                        forms_by_language["viet1250"].setdefault(major_cid, []).extend(proto_forms)


# ======================================================================
# Result container
# ======================================================================

@dataclass
class DatasetForms:
    """All forms from a single CLDF dataset, organised for triplet generation.

    Attributes:
        dataset_name: Short name of the originating Lexibank dataset
            (usually the repository directory name).
        forms_by_language: Nested mapping
            ``glottocode → cognateset_id → [Form, …]``.
        languages: ``glottocode → LanguageData`` (with *forms* list left
            empty; forms are accessed via *forms_by_language* instead).
        cognate_coverage: ``cognateset_id → {glottocodes with forms}``.
        proto_languages: Glottocodes identified as proto-languages.
        family: Best-guess language family for the whole dataset
            (from dataset metadata or the most frequent family value).
    """

    dataset_name: str
    forms_by_language: dict[str, dict[str, list[Form]]]
    languages: dict[str, LanguageData]
    cognate_coverage: dict[str, set[str]]
    proto_languages: set[str]
    family: str | None

    @property
    def num_forms(self) -> int:
        """Total number of forms across all languages and cognate sets."""
        return sum(
            len(forms)
            for by_cogset in self.forms_by_language.values()
            for forms in by_cogset.values()
        )

    @property
    def num_languages(self) -> int:
        """Number of distinct languages (glottocodes) with at least one form."""
        return len(self.forms_by_language)

    @property
    def num_cognate_sets(self) -> int:
        """Number of distinct cognate sets."""
        return len(self.cognate_coverage)


# ======================================================================
# Helpers
# ======================================================================

def _is_proto_language(name: str) -> bool:
    """Heuristic: does the language name suggest a proto-language?"""
    lower = name.lower()
    return lower.startswith("proto-") or lower.startswith("proto ")


def _find_cldf_metadata(dataset_dir: Path) -> Path | None:
    """Locate the CLDF metadata JSON inside a dataset directory.

    The metadata file is typically in a ``cldf/`` subdirectory and named
    ``Wordlist-metadata.json``, ``StructureDataset-metadata.json``,
    ``cldf-metadata.json``, etc.

    Returns:
        Path to the first metadata JSON found, or ``None``.
    """
    # Prefer cldf/ subdirectory (canonical Lexibank layout).
    cldf_dir = dataset_dir / "cldf"
    if cldf_dir.is_dir():
        for candidate in sorted(cldf_dir.glob("*-metadata.json")):
            return candidate
        # Also look for the simpler name
        simple = cldf_dir / "cldf-metadata.json"
        if simple.exists():
            return simple

    # Fall back: metadata JSON at the top level.
    for candidate in sorted(dataset_dir.glob("*-metadata.json")):
        return candidate

    return None


def _table_has_column(ds: Dataset, table_name: str, column_name: str) -> bool:
    """Check whether *table_name* in *ds* contains *column_name*.

    Returns ``False`` if the table itself is absent.
    """
    try:
        table = ds[table_name]
    except KeyError:
        return False

    return any(
        col.name == column_name or col.header == column_name
        for col in table.tableSchema.columns
    )


def _safe_str(value: Any) -> str:
    """Coerce a value to ``str``, returning ``""`` for ``None``."""
    return str(value) if value is not None else ""


def _safe_float(value: Any) -> float | None:
    """Try to parse a float, returning ``None`` on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


# ======================================================================
# Main loader
# ======================================================================

class CLDFLoader:
    """Load Lexibank CLDF datasets into :class:`DatasetForms` structures.

    Args:
        data_dir: Root directory containing one subdirectory per dataset.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_path: str | Path) -> DatasetForms | None:
        """Load a single CLDF dataset.

        Handles both cognate-set representations:

        * **Inline** — ``Cognateset_ID`` present directly in ``FormTable``.
        * **Separate** — a ``CognateTable`` that links ``Form_ID`` →
          ``Cognateset_ID``.

        Args:
            dataset_path: Path to the dataset directory (must contain a
                ``cldf/`` subdirectory with a CLDF metadata JSON).

        Returns:
            A populated :class:`DatasetForms`, or ``None`` when the dataset
            contains no cognate information.
        """
        dataset_path = Path(dataset_path)
        dataset_name = dataset_path.name

        metadata_path = _find_cldf_metadata(dataset_path)
        if metadata_path is None:
            logger.warning(
                "No CLDF metadata found in '%s' — skipping.", dataset_path,
            )
            return None

        try:
            ds = Dataset.from_metadata(metadata_path)
        except Exception:
            logger.exception(
                "Failed to load CLDF metadata '%s'.", metadata_path,
            )
            return None

        # ---- Determine cognate source ---------------------------------
        has_inline_cognates = _table_has_column(ds, "FormTable", "Cognateset_ID")
        has_cognate_table = self._has_cognate_table(ds)

        if not has_inline_cognates and not has_cognate_table:
            logger.info(
                "Dataset '%s' has no cognate information — skipping.",
                dataset_name,
            )
            return None

        # ---- Build language lookup ------------------------------------
        lang_lookup: dict[str, dict[str, Any]] = {}  # internal_id → info
        for lang in ds["LanguageTable"]:
            lid = _safe_str(lang.get("ID", ""))
            glottocode = _safe_str(lang.get("Glottocode", ""))
            name = _safe_str(lang.get("Name", ""))
            family = lang.get("Family") or lang.get("family") or None
            subgroup = lang.get("SubGroup") or lang.get("subgroup") or lang.get("Subgroup") or None
            lat = _safe_float(lang.get("Latitude"))
            lon = _safe_float(lang.get("Longitude"))

            if not lid:
                continue

            lang_lookup[lid] = {
                "glottocode": glottocode or lid,  # fallback
                "name": name,
                "family": _safe_str(family) if family else None,
                "subgroup": _safe_str(subgroup) if subgroup else None,
                "latitude": lat,
                "longitude": lon,
                "is_proto": _is_proto_language(name),
            }

        # ---- Build concept lookup -------------------------------------
        concept_lookup: dict[str, dict[str, Any]] = {}  # internal_id → info
        try:
            for concept in ds["ParameterTable"]:
                cid = _safe_str(concept.get("ID", ""))
                concepticon_id = concept.get("Concepticon_ID")
                gloss = (
                    concept.get("Concepticon_Gloss")
                    or concept.get("Name")
                    or ""
                )
                if cid:
                    concept_lookup[cid] = {
                        "concepticon_id": (
                            _safe_str(concepticon_id)
                            if concepticon_id
                            else None
                        ),
                        "gloss": _safe_str(gloss),
                    }
        except KeyError:
            logger.debug(
                "Dataset '%s' has no ParameterTable.", dataset_name,
            )

        # ---- Load cognate mapping (for CognateTable path) -------------
        cognate_map: dict[str, str] = {}  # form_id → cognateset_id
        if has_cognate_table and not has_inline_cognates:
            cognate_map = self._build_cognate_map(ds, dataset_name)

        # ---- Check which columns are available in FormTable -----------
        has_segments = _table_has_column(ds, "FormTable", "Segments")

        # ---- Iterate over forms ---------------------------------------
        forms_by_language: dict[str, dict[str, list[Form]]] = defaultdict(
            lambda: defaultdict(list)
        )
        cognate_coverage: dict[str, set[str]] = defaultdict(set)
        language_data: dict[str, LanguageData] = {}
        proto_languages: set[str] = set()
        family_counter: dict[str, int] = defaultdict(int)

        for row in ds["FormTable"]:
            form_id = _safe_str(row.get("ID", ""))
            lang_id = _safe_str(row.get("Language_ID", ""))
            param_id = _safe_str(row.get("Parameter_ID", ""))

            # Resolve language
            lang_info = lang_lookup.get(lang_id)
            if lang_info is None:
                continue
            glottocode = lang_info["glottocode"]

            # Extract segments / form string
            raw_segments = row.get("Segments") if has_segments else None
            raw_form = row.get("Form", "")

            # pycldf may return Segments as a list or as a space-separated string.
            if isinstance(raw_segments, list):
                raw_seg_list = [s for s in raw_segments if s]
            elif isinstance(raw_segments, str) and raw_segments.strip():
                raw_seg_list = raw_segments.strip().split()
            else:
                # Fall back to Form column.
                form_str = _safe_str(raw_form).strip()
                if not form_str:
                    continue  # skip empty forms
                raw_seg_list = form_str.split()

            clean_segments = []
            for s in raw_seg_list:
                # Lexibank segments often use SOURCE/TARGET for orthography/BIPA mapping.
                # We take the rightmost element to get the standardized BIPA representation.
                s = s.split("/")[-1]
                # Strip out uncertainty markers
                for char in "()[]~":
                    s = s.replace(char, "")
                if s:
                    clean_segments.append(s)

            segments = tuple(clean_segments)

            if not segments:
                continue  # skip empty

            # Resolve cognate set
            if has_inline_cognates:
                raw_cogset = row.get("Cognateset_ID")
                if raw_cogset is None or _safe_str(raw_cogset).strip() == "":
                    continue  # no cognate set assigned
                cognateset_id = f"{dataset_name}:{_safe_str(raw_cogset)}"
            else:
                raw_cogset_id = cognate_map.get(form_id)
                if raw_cogset_id is None:
                    continue  # no cognate assignment
                cognateset_id = raw_cogset_id  # already prefixed

            # Resolve concept
            concept_info = concept_lookup.get(param_id, {})
            concept_gloss = concept_info.get("gloss", param_id)
            concepticon_id = concept_info.get("concepticon_id")

            form = Form(
                form_id=form_id,
                language=glottocode,
                language_name=lang_info["name"],
                segments=segments,
                concept=concept_gloss,
                concepticon_id=concepticon_id,
                cognateset_id=cognateset_id,
                dataset=dataset_name,
            )

            forms_by_language[glottocode][cognateset_id].append(form)
            cognate_coverage[cognateset_id].add(glottocode)

            # Ensure language metadata entry exists
            if glottocode not in language_data:
                language_data[glottocode] = LanguageData(
                    glottocode=glottocode,
                    name=lang_info["name"],
                    family=lang_info["family"],
                    subgroup=lang_info["subgroup"],
                    latitude=lang_info["latitude"],
                    longitude=lang_info["longitude"],
                    is_proto=lang_info["is_proto"],
                )
                if lang_info["is_proto"]:
                    proto_languages.add(glottocode)
                if lang_info["family"]:
                    family_counter[lang_info["family"]] += 1

        # Apply dataset-specific manual overrides
        _apply_dataset_overrides(dataset_name, language_data, forms_by_language, proto_languages)

        if not forms_by_language:
            logger.info(
                "Dataset '%s' yielded no usable forms — skipping.",
                dataset_name,
            )
            return None

        # Determine dominant family
        dataset_family: str | None = None
        if family_counter:
            dataset_family = max(family_counter, key=family_counter.get)  # type: ignore[arg-type]

        return DatasetForms(
            dataset_name=dataset_name,
            forms_by_language=dict(forms_by_language),
            languages=language_data,
            cognate_coverage=dict(cognate_coverage),
            proto_languages=proto_languages,
            family=dataset_family,
        )

    def load_all_datasets(self) -> list[DatasetForms]:
        """Load every valid CLDF dataset found under :attr:`data_dir`.

        Datasets without cognate information are silently skipped.

        Returns:
            A list of :class:`DatasetForms`, one per successfully loaded
            dataset.
        """
        if not self.data_dir.exists():
            logger.warning("Data directory '%s' does not exist.", self.data_dir)
            return []

        results: list[DatasetForms] = []
        for subdir in sorted(self.data_dir.iterdir()):
            if not subdir.is_dir():
                continue
            loaded = self.load_dataset(subdir)
            if loaded is not None:
                results.append(loaded)
                logger.info(
                    "Loaded '%s': %d languages, %d forms, %d cognate sets.",
                    loaded.dataset_name,
                    loaded.num_languages,
                    loaded.num_forms,
                    loaded.num_cognate_sets,
                )

        logger.info("Loaded %d dataset(s) in total.", len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_cognate_table(ds: Dataset) -> bool:
        """Check whether the dataset includes a ``CognateTable``."""
        try:
            _ = ds["CognateTable"]
            return True
        except KeyError:
            return False

    @staticmethod
    def _build_cognate_map(ds: Dataset, dataset_name: str) -> dict[str, str]:
        """Build a form_id → cognateset_id mapping from ``CognateTable``.

        If a form appears in multiple cognate sets, only the first
        assignment is kept (matches typical Lexibank conventions).

        The cognate-set ID is prefixed with the dataset name to ensure
        dataset-scoping: ``"<dataset_name>:<raw_id>"``.
        """
        mapping: dict[str, str] = {}
        for row in ds["CognateTable"]:
            form_id = _safe_str(row.get("Form_ID", ""))
            raw_cogset = _safe_str(row.get("Cognateset_ID", ""))
            if form_id and raw_cogset:
                # A form may belong to multiple cognate sets; keep all
                # by not overwriting.  However, the main data structure
                # is keyed by (language, cognateset), so multiple
                # memberships are naturally handled by appending forms
                # to each cognate set.
                #
                # We store only the *first* mapping here.  For the rare
                # multi-cognate case, the form will appear under the
                # first cognate set only.  A more thorough implementation
                # could emit the form under every cognate set.
                if form_id not in mapping:
                    mapping[form_id] = f"{dataset_name}:{raw_cogset}"
        return mapping
