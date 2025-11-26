import argparse
import json
import sys
from pathlib import Path
from compile_dataset import create_datasets
from fine_tune import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full cognate reconstruction pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile the dataset, skip training",
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Skip dataset compilation, only train"
    )
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        sys.exit(1)


def main():
    args = parse_args()
    config = load_config(args.config)

    # Create necessary directories
    Path(config["dataset"]["output_train_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["dataset"]["output_test_path"]).mkdir(parents=True, exist_ok=True)
    Path(config["training"]["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # Run the full pipeline or selected parts
    if not args.train_only:
        print("\n=== Compiling Dataset ===\n")
        create_datasets(
            lexibank_path=config["dataset"]["lexibank_path"],
            test_folders=config["dataset"]["test_folders"],
            num_combinations=config["dataset"]["num_combinations"],
            concepts_per_text=config["dataset"]["concepts_per_text"],
            min_valid_cognates=config["dataset"]["min_valid_cognates"],
            langs_per_entry=config["dataset"]["langs_per_entry"],
            output_train_path=config["dataset"]["output_train_path"],
            output_test_path=config["dataset"]["output_test_path"],
        )

    if not args.compile_only:
        print("\n=== Training Model ===\n")
        train_model(args.config)

    print("\n=== Pipeline completed successfully ===\n")


if __name__ == "__main__":
    main()
