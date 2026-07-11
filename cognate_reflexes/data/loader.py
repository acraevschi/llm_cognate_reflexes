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
import unicodedata
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
            for variety_id, lang in language_data.items():
                if variety_id.endswith(f":{local_id}") or lang.name == local_id:
                    lang.glottocode = target_gc
                    lang.tree_glottocode = target_gc
                    if variety_id in forms_by_language:
                        forms_dict = forms_by_language[variety_id]
                        for forms in forms_dict.values():
                            for f in forms:
                                f.tree_glottocode = target_gc
                    break

    # 2. Cognate set linking overrides (e.g. for sidwellvietic)
    if ds_name == "sidwellvietic":
        param_cog_langs = {}
        proto_variety_ids = {
            variety_id
            for variety_id, lang in language_data.items()
            if lang.tree_glottocode == "viet1250"
        }
        for variety_id, forms_dict in forms_by_language.items():
            for cid, forms in forms_dict.items():
                if forms and forms[0].concepticon_id:
                    param = forms[0].concepticon_id
                    param_cog_langs.setdefault(param, {}).setdefault(cid, []).extend(forms)

        for param, cog_map in param_cog_langs.items():
            proto_cids = []
            attested_cids_with_counts = []
            for cid, forms in cog_map.items():
                is_proto = any(f.language in proto_variety_ids for f in forms)
                if is_proto:
                    proto_cids.append(cid)
                else:
                    attested_cids_with_counts.append((cid, len(forms)))
                    
            if proto_cids and attested_cids_with_counts:
                major_cid = max(attested_cids_with_counts, key=lambda x: x[1])[0]
                for proto_cid in proto_cids:
                    for proto_variety_id in proto_variety_ids:
                        if proto_cid not in forms_by_language.get(proto_variety_id, {}):
                            continue
                        proto_forms = forms_by_language[proto_variety_id].pop(proto_cid)
                        for f in proto_forms:
                            f.cognateset_id = major_cid
                        forms_by_language[proto_variety_id].setdefault(
                            major_cid, []
                        ).extend(proto_forms)


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
            ``variety_id → cognateset_id → [Form, …]``.
        languages: ``variety_id → LanguageData`` (with *forms* list left
            empty; forms are accessed via *forms_by_language* instead).
        cognate_coverage: ``cognateset_id → {variety_ids with forms}``.
        proto_languages: Variety IDs identified as proto-languages.
        historical_languages: Variety IDs explicitly marked historical by a
            source dataset.
        family: Best-guess language family for the whole dataset
            (from dataset metadata or the most frequent family value).
        source_path: Dataset root used to discover authoritative source trees.
    """

    dataset_name: str
    forms_by_language: dict[str, dict[str, list[Form]]]
    languages: dict[str, LanguageData]
    cognate_coverage: dict[str, set[str]]
    proto_languages: set[str]
    historical_languages: set[str]
    family: str | None
    source_path: Path | None = None

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
        """Number of distinct source varieties with at least one form."""
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


def _as_bool(value: Any) -> bool:
    """Parse common CLDF boolean encodings without guessing from names."""
    if isinstance(value, bool):
        return value
    return _safe_str(value).strip().lower() in {"1", "true", "yes"}


def _historical_date(row: dict[str, Any]) -> float | None:
    """Return a source-provided age before present, where one is available.

    IE-CoR uses several distribution-specific fields.  We deliberately retain
    only a single comparable point estimate here; uncertainty remains in the
    source table and lineage manifests provide the authoritative ancestry.
    """
    for column in (
        "normalMean",
        "NormalMean",
        "logNormalMean",
        "LogNormalMean",
    ):
        value = _safe_float(row.get(column))
        if value is not None:
            return value
    return None


def _tokens(value: Any) -> list[str]:
    """Return existing CLDF tokens without attempting raw-form tokenisation."""
    if isinstance(value, list):
        return [_safe_str(token).strip() for token in value if _safe_str(token).strip()]
    if isinstance(value, str):
        return [token for token in value.split() if token]
    return []


def _normalize_existing_tokens(tokens: list[str]) -> tuple[str, ...]:
    """Apply only lossless cross-dataset normalization to existing tokens.

    Raw ``Form`` values are deliberately not split here: many are orthographic
    rather than IPA.  Dataset-specific profiles must opt into that recovery
    path separately.
    """
    return tuple(unicodedata.normalize("NFC", token) for token in tokens)


def _clade_path(value: Any) -> tuple[str, ...]:
    """Coerce a CLDF list or delimited clade field into an ordered path."""
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = _safe_str(value).split(";")
    return tuple(_safe_str(part).strip() for part in parts if _safe_str(part).strip())


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
            source_glottocode = _safe_str(lang.get("Glottocode", ""))
            name = _safe_str(lang.get("Name", ""))
            family = lang.get("Family") or lang.get("family") or None
            subgroup = lang.get("SubGroup") or lang.get("subgroup") or lang.get("Subgroup") or None
            lat = _safe_float(lang.get("Latitude"))
            lon = _safe_float(lang.get("Longitude"))

            if not lid:
                continue

            lang_lookup[lid] = {
                # Forms are keyed by this value, never by a Glottocode.  A
                # source can legitimately assign the same Glottocode to a
                # modern variety and one or more of its historical stages.
                "variety_id": f"{dataset_name}:{lid}",
                "glottocode": source_glottocode or lid,
                "tree_glottocode": source_glottocode or lid,
                "name": name,
                "family": _safe_str(family) if family else None,
                "subgroup": _safe_str(subgroup) if subgroup else None,
                "latitude": lat,
                "longitude": lon,
                "is_proto": _is_proto_language(name),
                "is_historical": _as_bool(
                    lang.get("historical", lang.get("Historical"))
                ),
                "date_before_present": _historical_date(lang),
                "clade_path": _clade_path(
                    lang.get("Clade")
                    or lang.get("clade")
                    or lang.get("SubGroup")
                    or lang.get("subgroup")
                ),
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
        has_phonemic_segments = _table_has_column(
            ds, "FormTable", "Phonemic_Segments"
        )

        # ---- Iterate over forms ---------------------------------------
        forms_by_language: dict[str, dict[str, list[Form]]] = defaultdict(
            lambda: defaultdict(list)
        )
        cognate_coverage: dict[str, set[str]] = defaultdict(set)
        language_data: dict[str, LanguageData] = {}
        proto_languages: set[str] = set()
        historical_languages: set[str] = set()
        family_counter: dict[str, int] = defaultdict(int)

        for row in ds["FormTable"]:
            form_id = _safe_str(row.get("ID", ""))
            lang_id = _safe_str(row.get("Language_ID", ""))
            param_id = _safe_str(row.get("Parameter_ID", ""))

            # Resolve language
            lang_info = lang_lookup.get(lang_id)
            if lang_info is None:
                continue
            variety_id = lang_info["variety_id"]

            # Prefer standard CLDF Segments.  IE-CoR exposes a second,
            # profile-tokenised Phonemic_Segments field which is a safe
            # fallback.  Never split a raw Form generically: it can be
            # orthographic and would introduce systematic pseudo-IPA noise.
            segment_source = "segments"
            raw_seg_list = _tokens(row.get("Segments")) if has_segments else []
            if not raw_seg_list and has_phonemic_segments:
                raw_seg_list = _tokens(row.get("Phonemic_Segments"))
                segment_source = "phonemic_segments"
            segments = _normalize_existing_tokens(raw_seg_list)

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
                language=variety_id,
                language_name=lang_info["name"],
                segments=segments,
                concept=concept_gloss,
                concepticon_id=concepticon_id,
                cognateset_id=cognateset_id,
                dataset=dataset_name,
                tree_glottocode=lang_info["tree_glottocode"],
                segment_source=segment_source,
            )

            forms_by_language[variety_id][cognateset_id].append(form)
            cognate_coverage[cognateset_id].add(variety_id)

            # Ensure language metadata entry exists
            if variety_id not in language_data:
                language_data[variety_id] = LanguageData(
                    glottocode=lang_info["glottocode"],
                    name=lang_info["name"],
                    family=lang_info["family"],
                    subgroup=lang_info["subgroup"],
                    latitude=lang_info["latitude"],
                    longitude=lang_info["longitude"],
                    is_proto=lang_info["is_proto"],
                    variety_id=variety_id,
                    tree_glottocode=lang_info["tree_glottocode"],
                    is_historical=lang_info["is_historical"],
                    date_before_present=lang_info["date_before_present"],
                    clade_path=lang_info["clade_path"],
                )
                if lang_info["is_proto"]:
                    proto_languages.add(variety_id)
                if lang_info["is_historical"]:
                    historical_languages.add(variety_id)
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
            historical_languages=historical_languages,
            family=dataset_family,
            source_path=dataset_path,
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
