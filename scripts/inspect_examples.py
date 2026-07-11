#!/usr/bin/env python3
"""Interactive script to manually inspect generated examples.

Usage:
    python scripts/inspect_examples.py --task cognate_reflex
    python scripts/inspect_examples.py --task reconstruction
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to sys.path so we can import cognate_reflexes without installing
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from cognate_reflexes import Config, ExampleGenerator, TextFormatter

def main():
    parser = argparse.ArgumentParser(description="Inspect examples interactively.")
    parser.add_argument("--task", choices=["cognate_reflex", "reconstruction"], default="cognate_reflex")
    parser.add_argument("--data-dir", default="./data/lexibank")
    parser.add_argument("--glottolog-dir", default="./data/glottolog")
    args = parser.parse_args()

    # Disable extensive logging to keep the console clean
    logging.getLogger("cognate_reflexes").setLevel(logging.WARNING)

    config = Config(
        task=args.task,
        data_dir=args.data_dir,
        glottolog_dir=args.glottolog_dir,
        max_triplets_per_dataset=50  # Limit to avoid unnecessary computation
    )
    
    gen = ExampleGenerator(config=config)
    formatter = TextFormatter(show_language_names=True)
    
    print(f"Generating examples for task: {args.task}...")
    print("Press Enter to see the next example. Type 'q' to quit.")
    
    try:
        for i, example in enumerate(gen.generate()):
            print("\n" + "="*80)
            print(f"Example #{i+1}")
            print(f"Dataset: {example.metadata.source_dataset} | Family: {example.metadata.language_family}")
            print(f"Glottocodes: {example.metadata.glottocodes}")
            print("="*80)
            
            input_text, target_text = formatter.format_example(example)
            print(input_text)
            print("\n[GROUND TRUTH]")
            print(target_text)
            print("-" * 80)
            
            cmd = input("Press Enter for next, 'q' to quit: ")
            if cmd.strip().lower() == 'q':
                break
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
