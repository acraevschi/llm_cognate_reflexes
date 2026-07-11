#!/usr/bin/env python3
"""Generate training triplets from Lexibank data.

Usage examples::

    # Cognate reflex triplets (Stage 1)
    python scripts/generate_triplets.py --task cognate_reflex -o triplets_reflex.jsonl

    # Proto-language reconstruction triplets (Stage 2)
    python scripts/generate_triplets.py --task reconstruction -o triplets_recon.jsonl

    # Show dataset statistics without generating
    python scripts/generate_triplets.py --stats-only

    # Customise parameters
    python scripts/generate_triplets.py \\
        --task cognate_reflex \\
        --min-cognates 20 \\
        --max-cognates 80 \\
        --mask-ratio 0.4 \\
        --max-edge-distance 6 \\
        -o output.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path so we can import cognate_reflexes without installing
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cognate_reflexes import Config, ExampleGenerator
from cognate_reflexes.formatting.formatter import TextFormatter
from cognate_reflexes.formatting.serializer import ExampleSerializer


def _print_stats(stats: dict[str, object]) -> None:
    """Pretty-print dataset statistics."""
    print("\n" + "=" * 60)
    print("  DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total datasets found:          {stats['num_datasets']}")
    print(f"  Datasets with cognate sets:    {stats['num_datasets_with_cognates']}")
    print(f"  Datasets with proto-forms:     {stats['num_datasets_with_proto']}")
    print(f"  Total languages:               {stats['total_languages']}")
    print(f"  Total forms:                   {stats['total_forms']}")
    families = stats.get("families", [])
    print(f"  Language families:             {len(families)}")
    if families:
        print("  Families:")
        for fam in families[:20]:
            print(f"    - {fam}")
        if len(families) > 20:
            print(f"    … and {len(families) - 20} more")
    print("=" * 60 + "\n")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training triplets from Lexibank data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--task",
        choices=["cognate_reflex", "reconstruction"],
        default="cognate_reflex",
        help="Which task to generate triplets for (default: cognate_reflex).",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/lexibank",
        help="Directory containing Lexibank CLDF datasets.",
    )
    parser.add_argument(
        "--glottolog-dir",
        default="./data/glottolog",
        help="Path to cloned Glottolog repository.",
    )
    parser.add_argument(
        "--output", "-o",
        default="triplets.jsonl",
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--min-cognates",
        type=int,
        default=15,
        help="Minimum shared cognate sets per triplet (default: 15).",
    )
    parser.add_argument(
        "--max-cognates",
        type=int,
        default=100,
        help="Maximum cognate sets per triplet (default: 100).",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.3,
        help="Fraction of target forms to mask (default: 0.3).",
    )
    parser.add_argument(
        "--max-branch-length",
        type=float,
        default=None,
        help="Max total branch length between languages in a triplet.",
    )
    parser.add_argument(
        "--max-edge-distance",
        type=int,
        default=None,
        help="Max edge count between languages (fallback for no branch lengths).",
    )
    parser.add_argument(
        "--include-glosses",
        action="store_true",
        default=True,
        help="Include concept glosses in output (default: True).",
    )
    parser.add_argument(
        "--no-glosses",
        action="store_true",
        help="Disable concept glosses.",
    )
    parser.add_argument(
        "--max-polytomy-resolutions",
        type=int,
        default=50,
        help="Cap on binary resolutions per polytomy node (default: 50).",
    )
    parser.add_argument(
        "--max-triplets-per-dataset",
        type=int,
        default=None,
        help="Maximum triplets to generate per dataset (default: None).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print dataset statistics and exit (no generation).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers for multiprocessing (default: 4).",
    )

    args = parser.parse_args()

    # Setup logging.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)-30s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = Config(
        data_dir=args.data_dir,
        glottolog_dir=args.glottolog_dir,
        task=args.task,
        min_cognates=args.min_cognates,
        max_cognates=args.max_cognates,
        mask_ratio=args.mask_ratio,
        max_branch_length=args.max_branch_length,
        max_edge_distance=args.max_edge_distance,
        include_glosses=not args.no_glosses,
        max_polytomy_resolutions=args.max_polytomy_resolutions,
        max_triplets_per_dataset=args.max_triplets_per_dataset,
        seed=args.seed,
    )

    gen = ExampleGenerator(config=config)

    if args.stats_only:
        stats = gen.stats()
        _print_stats(stats)
        return

    # Generate and serialise.
    formatter = TextFormatter(
        include_glosses=config.include_glosses,
        show_language_names=config.show_language_names,
    )
    serializer = ExampleSerializer(formatter=formatter)

    start = time.time()
    count = serializer.write_jsonl(gen.generate(workers=args.workers), args.output)
    elapsed = time.time() - start

    print(f"\nGenerated {count:,} triplets in {elapsed:.1f}s")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
