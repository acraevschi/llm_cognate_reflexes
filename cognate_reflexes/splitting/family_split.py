"""Helper utilities for resolving language families robustly."""

from __future__ import annotations

import logging

try:
    from pyglottolog import Glottolog
except ImportError:
    Glottolog = None

logger = logging.getLogger(__name__)

# Hardcoded overrides for datasets lacking reliable tree glottocodes where
# standard fallbacks fail to resolve the top-level family.
MANUAL_OVERRIDES = {
    "tuled": ("Tupian", "tupi1275"),
    "mcd": ("Austronesian", "aust1307"),
}

class FamilyResolver:
    """Resolves canonical language families from Glottolog with fallbacks."""
    
    def __init__(self, glottolog_path: str | None = None):
        self.glottolog = Glottolog(glottolog_path) if glottolog_path and Glottolog else None
        self._cache: dict[str, tuple[str, str] | None] = {}
        
    def resolve_glottocode(self, code: str | None) -> tuple[str, str] | None:
        """Returns (family_name, family_id) for a given glottocode."""
        if not code:
            return None
            
        if code in self._cache:
            return self._cache[code]
            
        if not self.glottolog:
            return None
            
        try:
            lang = self.glottolog.languoid(code)
            if lang:
                if lang.lineage:
                    result = (lang.lineage[0].name, lang.lineage[0].id)
                else:
                    result = (lang.name, lang.id)
                self._cache[code] = result
                return result
        except Exception as e:
            logger.debug(f"Failed to resolve glottocode {code}: {e}")
            
        self._cache[code] = None
        return None

    def resolve_record(self, record: dict) -> tuple[str, str]:
        """Resolve the top-level canonical family for a JSONL record."""
        target = record.get("raw", {}).get("target", {})
        inputs = record.get("raw", {}).get("inputs", [])
        meta = record.get("metadata", {})
        
        source_dataset = meta.get("source_dataset")
        
        # 1. Manual Overrides (by dataset)
        if source_dataset in MANUAL_OVERRIDES:
            return MANUAL_OVERRIDES[source_dataset]
            
        # 2. Try the target's tree_glottocode or glottocode
        code = target.get("tree_glottocode") or target.get("glottocode")
        resolved = self.resolve_glottocode(code)
        if resolved:
            return resolved
            
        # 3. Fallback: Check if inputs unambiguously point to one family
        resolved_inputs = set()
        for inp in inputs:
            icode = inp.get("tree_glottocode") or inp.get("glottocode")
            r = self.resolve_glottocode(icode)
            if r:
                resolved_inputs.add(r)
                
        if len(resolved_inputs) == 1:
            return list(resolved_inputs)[0]
            
        # 4. Fallback: Raw family field if consistent
        target_family = target.get("family")
        if target_family:
            return (target_family, target_family.lower().replace(" ", "_"))
            
        # Unresolved
        return ("UNKNOWN", "unknown")
