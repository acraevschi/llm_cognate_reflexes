"""Tests for family splitting and fallbacks."""

import json
from cognate_reflexes.splitting.family_split import FamilyResolver

def test_manual_override_fallback():
    resolver = FamilyResolver()
    # Mock record for MCD
    mcd_record = {
        "metadata": {"source_dataset": "mcd"},
        "raw": {"target": {"glottocode": None}, "inputs": []}
    }
    family_name, family_id = resolver.resolve_record(mcd_record)
    assert family_name == "Austronesian"
    
    # Mock record for tuled
    tuled_record = {
        "metadata": {"source_dataset": "tuled"},
        "raw": {"target": {"glottocode": None}, "inputs": []}
    }
    family_name, family_id = resolver.resolve_record(tuled_record)
    assert family_name == "Tupian"

def test_fallback_inputs_unambiguous():
    resolver = FamilyResolver()
    
    # Mock pyglottolog responses using cache manipulation
    resolver._cache["code1"] = ("Indo-European", "indo1319")
    resolver._cache["code2"] = ("Indo-European", "indo1319")
    
    record = {
        "metadata": {"source_dataset": "some_dataset"},
        "raw": {
            "target": {"glottocode": None},
            "inputs": [
                {"glottocode": "code1"},
                {"glottocode": "code2"}
            ]
        }
    }
    family_name, family_id = resolver.resolve_record(record)
    assert family_name == "Indo-European"

def test_fallback_target_family_field():
    resolver = FamilyResolver()
    record = {
        "metadata": {"source_dataset": "some_dataset"},
        "raw": {
            "target": {
                "glottocode": None,
                "family": "Dravidian"
            },
            "inputs": []
        }
    }
    family_name, family_id = resolver.resolve_record(record)
    assert family_name == "Dravidian"

def test_unresolved_fallback():
    resolver = FamilyResolver()
    record = {
        "metadata": {"source_dataset": "some_dataset"},
        "raw": {
            "target": {
                "glottocode": None,
            },
            "inputs": []
        }
    }
    family_name, family_id = resolver.resolve_record(record)
    assert family_name == "UNKNOWN"
