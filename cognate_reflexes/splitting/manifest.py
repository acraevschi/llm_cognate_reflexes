"""Manifest generation for family-disjoint splits."""

import json
from pathlib import Path
from typing import Any
import datetime

class SplitManifest:
    def __init__(self, glottolog_version: str, seed: int | None = None):
        self.glottolog_version = glottolog_version
        self.seed = seed
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.family_assignments: dict[str, str] = {}  # family_name -> split
        self.stats: dict[str, dict[str, Any]] = {
            "train": {"stage1_count": 0, "stage2_count": 0},
            "dev": {"stage1_count": 0, "stage2_count": 0},
            "test1": {"stage1_count": 0, "stage2_count": 0},
            "test2": {"stage1_count": 0, "stage2_count": 0},
        }
        self.fallbacks_used = []
        self.input_checksums = {}
        
    def assign_family(self, family_name: str, split: str):
        self.family_assignments[family_name] = split
        
    def get_assignment(self, family_name: str) -> str:
        return self.family_assignments.get(family_name, "train")
        
    def record_stat(self, split: str, stage: int):
        stage_key = f"stage{stage}_count"
        if split in self.stats:
            self.stats[split][stage_key] += 1
            
    def save(self, output_path: str | Path):
        data = {
            "timestamp": self.timestamp,
            "glottolog_version": self.glottolog_version,
            "seed": self.seed,
            "family_assignments": self.family_assignments,
            "stats": self.stats,
            "input_checksums": self.input_checksums,
            "fallbacks_used": self.fallbacks_used
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
