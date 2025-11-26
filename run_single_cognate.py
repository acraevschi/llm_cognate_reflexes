import argparse
import json
import sys
from pathlib import Path
from fine_tune_single_cognate import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full cognate reconstruction pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
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

    print("\n=== Training Model ===\n")
    train_model(args.config)

    print("\n=== Pipeline completed successfully ===\n")


if __name__ == "__main__":
    main()
