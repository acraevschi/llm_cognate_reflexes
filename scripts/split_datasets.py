"""Script to stream JSONL examples and split them into family-disjoint sets."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from cognate_reflexes.splitting.family_split import FamilyResolver
from cognate_reflexes.splitting.manifest import SplitManifest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_checksum(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def split_file(
    input_path: Path,
    output_prefix: Path,
    stage: int,
    resolver: FamilyResolver,
    manifest: SplitManifest,
    dev_families: set[str],
    test1_families: set[str],
    test2_families: set[str],
):
    if not input_path.exists():
        logger.warning(f"File not found: {input_path}")
        return

    manifest.input_checksums[input_path.name] = compute_checksum(input_path)
    
    train_path = output_prefix.parent / f"{output_prefix.name}_train.jsonl"
    dev_path = output_prefix.parent / f"{output_prefix.name}_dev.jsonl"
    test1_path = output_prefix.parent / f"{output_prefix.name}_test1.jsonl"
    test2_path = output_prefix.parent / f"{output_prefix.name}_test2.jsonl"
    
    files = {
        "train": open(train_path, 'w', encoding='utf-8'),
        "dev": open(dev_path, 'w', encoding='utf-8'),
        "test1": open(test1_path, 'w', encoding='utf-8'),
        "test2": open(test2_path, 'w', encoding='utf-8'),
    }
    
    with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            record = json.loads(line)
            family_name, _ = resolver.resolve_record(record)
            
            if family_name in dev_families:
                split = "dev"
            elif family_name in test1_families:
                split = "test1"
            elif family_name in test2_families:
                split = "test2"
            else:
                split = "train"
                
            manifest.assign_family(family_name, split)
            manifest.record_stat(split, stage)
            
            # Write to the assigned split
            files[split].write(line)
            
    for f in files.values():
        f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--glottolog-dir", type=Path, default=Path("data/glottolog"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    resolver = FamilyResolver(str(args.glottolog_dir))
    
    manifest = SplitManifest(glottolog_version="pinned_local", seed=args.seed)
    
    dev_families = {"Austronesian"}
    test1_families = {"Arawakan"}
    test2_families = {"Austroasiatic"}
    
    logger.info("Splitting Stage 1...")
    split_file(
        args.data_dir / "stage1_cognate_reflex.jsonl",
        args.data_dir / "stage1_cognate_reflex",
        stage=1,
        resolver=resolver,
        manifest=manifest,
        dev_families=dev_families,
        test1_families=test1_families,
        test2_families=test2_families,
    )

    logger.info("Splitting Stage 2...")
    split_file(
        args.data_dir / "stage2_reconstruction.jsonl",
        args.data_dir / "stage2_reconstruction",
        stage=2,
        resolver=resolver,
        manifest=manifest,
        dev_families=dev_families,
        test1_families=test1_families,
        test2_families=test2_families,
    )
    
    manifest_path = args.data_dir / "split_manifest.json"
    manifest.save(manifest_path)
    logger.info(f"Done. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    main()
