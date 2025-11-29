import os
from datasets import Dataset
from tqdm import tqdm
import argparse
import sys
import json
import numpy as np
from multiprocessing import Pool, cpu_count

# --- Helper functions for multiprocessing ---
from cognate_utils import process_family_folder

DEFAULT_CONFIG = {
    "test_folders": [
        "mannburmish",
        "gerarditupi",
        "savelyevturkic",
        "ratcliffearabic",
        "walworthpolynesian",
    ],
    # NEW: Specific Validation Folders
    "val_folders": [
        "starostintujia",
        "sagartst",
        "saenkoromance"
    ],
    "num_evidence_sets": 25,  
    "num_combinations": 5,  
    "min_valid_cognates": 5,  
    "lexibank_path": "lexibank",
    "langs_per_entry": 3,  
    "output_train_path": "datasets/single_cognate_anon/hf_cognates_dataset",
    "output_eval_path": "datasets/single_cognate_anon/hf_cognates_eval_dataset",
    "output_test_path": "datasets/single_cognate_anon/hf_cognates_test_dataset",
    "test_split_ratio": 0.25, 
}


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


# --- Main Execution ---


def create_datasets(
    lexibank_path,
    test_folders,
    val_folders, # NEW ARGUMENT
    num_combinations,
    num_evidence_sets,
    min_valid_cognates,
    langs_per_entry,
    output_train_path="hf_cognates_dataset",
    output_val_path="hf_cognates_val_dataset", # NEW ARGUMENT
    output_test_path="hf_cognates_test_dataset",
    test_split_ratio=0.1,  
    num_lang_fams=None,
    num_proc=4,
):

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_val_path, exist_ok=True) # Ensure val path exists
    os.makedirs(output_test_path, exist_ok=True)

    folders = [f for f in os.listdir(lexibank_path) if not f.endswith(".tsv")]

    if num_lang_fams:
        folders = folders[:num_lang_fams]

    # Prepare arguments for parallel execution
    tasks = []
    for folder in folders:
        is_test_folder = folder in test_folders
        is_val_folder = folder in val_folders # Check if val
        
        task_args = (
            folder,
            lexibank_path,
            is_test_folder,
            is_val_folder, # Pass val flag
            num_combinations,
            num_evidence_sets,
            min_valid_cognates,
            langs_per_entry,
            test_split_ratio,
        )
        tasks.append(task_args)

    print(f"Starting parallel processing on {num_proc} cores...")
    print(f"Validation Strategy: Strict OOD. Test & Val folders will be unseen in Training.")

    global_train_entries = []
    global_val_entries = [] # NEW LIST
    global_test_entries = []

    # Parallel Execution
    with Pool(processes=num_proc) as pool:
        results = list(
            tqdm(
                pool.imap(process_family_folder, tasks),
                total=len(tasks),
                desc="Processing Families",
            )
        )

    # Aggregate results
    print("Aggregating results...")
    for train_chunk, val_chunk, test_chunk in results: # Unpack 3 lists
        global_train_entries.extend(train_chunk)
        global_val_entries.extend(val_chunk)
        global_test_entries.extend(test_chunk)

    print(f"Total Generated Train Examples: {len(global_train_entries)}")
    print(f"Total Generated Val Examples:   {len(global_val_entries)}")
    print(f"Total Generated Test Examples:  {len(global_test_entries)}")

    if len(global_train_entries) == 0:
        print("WARNING: No training examples generated.")

    # Create HF Datasets
    hf_dataset = Dataset.from_list(global_train_entries)
    hf_val_dataset = Dataset.from_list(global_val_entries)
    hf_test_dataset = Dataset.from_list(global_test_entries)

    dataset_name = f"{langs_per_entry}langs_{num_evidence_sets}evidence"

    hf_dataset.save_to_disk(
        f"{output_train_path}/{dataset_name}", max_shard_size="25MB"
    )
    hf_val_dataset.save_to_disk(
        f"{output_val_path}/{dataset_name}", max_shard_size="25MB"
    )
    hf_test_dataset.save_to_disk(
        f"{output_test_path}/{dataset_name}", max_shard_size="25MB"
    )

    return hf_dataset, hf_val_dataset, hf_test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile single-cognate HF datasets.")
    parser.add_argument(
        "-c",
        "--config",
        required=False,
        help="Path to JSON config file (default: config.json)",
    )
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = DEFAULT_CONFIG

    create_datasets(
        lexibank_path=config["lexibank_path"],
        test_folders=config["test_folders"],
        val_folders=config.get("val_folders", []), # Safely get val_folders
        num_combinations=config["num_combinations"],
        num_evidence_sets=config["num_evidence_sets"],
        min_valid_cognates=config["min_valid_cognates"],
        langs_per_entry=config["langs_per_entry"],
        output_train_path=config["output_train_path"],
        output_val_path=config["output_eval_path"],
        output_test_path=config["output_test_path"],
        test_split_ratio=config["test_split_ratio"],
        num_proc=10,
    )