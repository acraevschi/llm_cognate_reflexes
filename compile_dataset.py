import os
import pandas as pd
import random
from datasets import Dataset
from tqdm import tqdm
import itertools
import math
import argparse
import sys
import json


# from lexibank_prep.lexibank_help import check_glotto_coverage
# from glotto_trees.get_newick import get_phylogenetic_tree

# Define default values
DEFAULT_CONFIG = {
    "test_folders": [
        "mannburmish",  # tonal, contains multiword forms, has a Glottocode, South-East Asia
        "gerarditupi",  # non-tonal, has a Glottocode, South/Centre of South America
        "savelyevturkic",  # non-tonal, vowel harmony, has a Glottocode, Eurasia
        "ratcliffearabic",  # non-tonal, fairly complex phonotactics, has a Glottocode, Middle East/Africa
        "walworthpolynesian",  # tonal, simple phonotactics occasional multimorphemic forms, has a Glottocode, Polynesia
    ],
    "concepts_per_text": 25,
    "num_combinations": 50,
    "min_valid_cognates": 8,
    "lexibank_path": "lexibank",
    "langs_per_entry": 3,
    "output_train_path": "hf_cognates_dataset",
    "output_test_path": "hf_cognates_test_dataset",
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


def mask_values(row, prop_mask=0.25, missing_threshold=0.5):
    """Same as original but works on subset of columns"""
    columns = row.index[1:]  # Exclude Cognate_ID
    non_missing_columns = [col for col in columns if row[col] != "-"]

    num_available = len(non_missing_columns)
    if (num_available == 0) or (num_available < (len(columns) * missing_threshold)):
        return []

    num_to_mask = min(max(1, int(len(columns) * prop_mask)), num_available)
    mask_indices = random.sample(non_missing_columns, num_to_mask)

    row_masked = row.copy()
    for col in mask_indices:
        row_masked[col] = "?"

    return row_masked


def clean_form(form):
    """Unchanged from original"""
    if pd.isna(form) or form == "":
        return form

    form_lst = form.split()
    for i, form in enumerate(form_lst):
        if "/" in form:
            form_lst[i] = form.split("/")[0]
    return "".join(form_lst).replace("+", " ")


def format_example(row, masked_row):
    """Modified to return only the reconstructed forms in the target"""
    cognate_id = row["Cognate_ID"]
    cognate_id = cognate_id.replace("?", "")
    cognate_id = cognate_id.split(",")[-1]
    cognate_id = cognate_id.split("-")[-1]
    # Get only the languages present in this chunk
    langs = row.index[1:]

    # Create input text with all forms (some masked)
    forms_masked = [f"{lang} = {clean_form(masked_row[lang])}" for lang in langs]
    input_text = f"<{cognate_id}>\n" + "\n".join(forms_masked) + f"\n</{cognate_id}>\n"

    # Find which languages were masked (have "?" in masked_row)
    masked_langs = [lang for lang in langs if masked_row[lang] == "?"]

    # Create target text with only the reconstructed forms
    reconstructed_forms = [f"{lang} = {clean_form(row[lang])}" for lang in masked_langs]
    target_text = (
        f"<{cognate_id}>\n" + "\n".join(reconstructed_forms) + f"\n</{cognate_id}>\n"
    )

    return input_text, target_text


def has_enough_languages(series, langs_per_entry, threshold=0.6):
    """Check if the series has enough languages with non-missing values."""
    no_heading = series.iloc[1:]  # Skip the first element (Cognate_ID)
    non_missing_count = sum(1 for value in no_heading if value != "-")
    return non_missing_count >= langs_per_entry * threshold


def create_datasets(
    lexibank_path,
    test_folders,
    num_combinations,
    concepts_per_text,
    min_valid_cognates,
    langs_per_entry,
    output_train_path="hf_cognates_dataset",
    output_test_path="hf_cognates_test_dataset",
):
    lexibank_folders_path = lexibank_path
    folders = os.listdir(lexibank_folders_path)
    dataset_entries = []
    dataset_test_entries = []
    test_combinations = set()

    # Ensure output directories exist
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    for folder in tqdm(folders, desc="Processing folders"):
        test_data = folder in test_folders

        if folder.endswith(".tsv"):
            continue

        data_path = f"lexibank/{folder}/wide_df.tsv"
        langs_path = f"lexibank/{folder}/languages.csv"

        try:
            data = pd.read_csv(data_path, sep="\t", encoding="utf-8").fillna("-")
            langs = pd.read_csv(langs_path, encoding="utf-8")
        except FileNotFoundError:
            continue

        # Column renaming
        # column_mapping = {
        #     langs["ID"][i]: (
        #         f"{langs['ID'][i]}:{langs['Glottocode'][i]}"
        #         if pd.notna(langs["Glottocode"][i])
        #         else langs["ID"][i]
        #     )
        #     for i in range(len(langs))
        # }
        # data.columns = [data.columns[0]] + [
        #     column_mapping.get(col, col) for col in data.columns[1:]
        # ]

        # Get language columns and check if there are at least 3
        # Get language columns and check if there are at least 3
        lang_columns = data.columns[1:].tolist()
        if len(lang_columns) < 3:
            continue  # Skip folders with fewer than 3 languages

        # Process combinations based on dataset type
        if test_data:
            # For test folders, randomly sample languages to make it deterministic
            # but avoid having to generate all combinations
            selected_combinations = []
            combinations_generator = itertools.combinations(
                lang_columns, langs_per_entry
            )

            # Take first num_combinations from generator
            for _ in range(num_combinations):
                try:
                    combo = next(combinations_generator)
                    selected_combinations.append(combo)
                    # Add to global test combinations (sorted to avoid order variations)
                    test_combinations.add(tuple(sorted(combo)))
                except StopIteration:
                    break  # In case there are fewer than num_combinations
        else:
            # For non-test folders: filter out test combinations as we go
            selected_combinations = []
            combinations_generator = itertools.combinations(
                lang_columns, langs_per_entry
            )

            # Collect num_combinations that aren't in test_combinations
            while len(selected_combinations) < num_combinations:
                try:
                    combo = next(combinations_generator)
                    if tuple(sorted(combo)) not in test_combinations:
                        selected_combinations.append(combo)
                except StopIteration:
                    break  # Stop if we run out of combinations

        # Process each selected combination
        for group in selected_combinations:
            group_columns = ["Cognate_ID"] + list(group)
            data_subset = data[group_columns]

            # Generate Newick tree (placeholder for actual implementation)
            # newick = ""
            # try:
            #     glottocodes = [col.split(":")[1] if ":" in col else "" for col in group]
            #     newick = get_phylogenetic_tree(glottocodes)
            # except:
            #     newick = ""

            # Shuffle the data
            data_subset = data_subset.sample(frac=1).reset_index(drop=True)
            n = len(data_subset)

            # Determine chunks such that each has equal size and no more than concepts_per_text rows.
            if n <= concepts_per_text:
                chunks = [data_subset]
            else:
                num_splits = math.ceil(n / concepts_per_text)

                chunk_size = n // num_splits
                chunks = [
                    data_subset[i * chunk_size : (i + 1) * chunk_size]
                    for i in range(num_splits)
                ]

            # Process each chunk separately
            for chunk in chunks:
                input_text = "<Cognates>\n"
                target_text = "<Reconstructed Cognates>\n"
                valid_cognate = 0

                # CHANGE: Pre-filter valid rows before processing
                valid_rows = []
                for i in range(len(chunk)):
                    row = chunk.iloc[i]
                    if has_enough_languages(row, langs_per_entry, threshold=1):
                        valid_rows.append(row)

                # CHANGE: Skip if not enough valid cognates
                if (
                    len(valid_rows) < min_valid_cognates
                ):  # Minimum number of cognates per entry
                    continue

                # CHANGE: Process up to concepts_per_text valid cognates
                for row in valid_rows[:concepts_per_text]:
                    valid_cognate += 1
                    masked_row = mask_values(row)

                    if len(masked_row) > 0:
                        cog_input, cog_output = format_example(row, masked_row)
                    else:
                        cog_input, cog_output = format_example(row, row)

                    input_text += cog_input
                    target_text += cog_output

                # Only add entries with multiple cognates
                if valid_cognate < min_valid_cognates:  # Minimum cognates threshold
                    continue

                input_text += "</Cognates>\n"
                target_text += "</Reconstructed Cognates>"

                # Add to appropriate dataset
                if test_data:
                    dataset_test_entries.append(
                        {"input": input_text, "output": target_text}
                    )
                else:
                    dataset_entries.append({"input": input_text, "output": target_text})

    # Create final datasets
    hf_dataset = Dataset.from_list(dataset_entries)
    hf_test_dataset = Dataset.from_list(dataset_test_entries)

    # # Save the dataset
    hf_dataset.save_to_disk(
        f"{output_train_path}/{concepts_per_text}concepts_min{min_valid_cognates}_{num_combinations}combs",
        max_shard_size="50MB",
    )
    hf_test_dataset.save_to_disk(
        f"{output_test_path}/{concepts_per_text}concepts_min{min_valid_cognates}_{num_combinations}combs",
        max_shard_size="50MB",
    )
    print("Datasets saved successfully!")

    return hf_dataset, hf_test_dataset


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compile cognate dataset")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file (JSON)",
    )
    parser.add_argument(
        "--concepts-per-text",
        type=int,
        help="Maximal number of concepts per one input/output",
    )
    parser.add_argument(
        "--num-combinations",
        type=int,
        help="Number of combinations to generate for each dataset",
    )
    parser.add_argument(
        "--min-valid-cognates",
        type=int,
        help="Minimum number of valid cognates per entry",
    )
    parser.add_argument(
        "--lexibank-path",
        type=str,
        help="Path to the Lexibank folder",
    )
    parser.add_argument(
        "--langs-per-entry",
        type=int,
        help="Number of languages per entry",
    )
    parser.add_argument(
        "--output-train-path",
        type=str,
        help="Path to save the training dataset",
    )
    parser.add_argument(
        "--output-test-path",
        type=str,
        help="Path to save the test dataset",
    )

    args = parser.parse_args()

    # Determine which configuration to use
    config = DEFAULT_CONFIG.copy()

    if args.config:
        # Check if any other argument is provided
        other_args = [
            args.concepts_per_text,
            args.num_combinations,
            args.min_valid_cognates,
            args.lexibank_path,
            args.langs_per_entry,
            args.output_train_path,
            args.output_test_path,
        ]

        if any(arg is not None for arg in other_args):
            print("Error: Cannot provide both config file and command-line arguments.")
            print("Please use either --config OR individual parameters.")
            sys.exit(1)

        # Load configuration from file
        if args.config:
            config.update(load_config(args.config))
        else:
            # Update config with command-line arguments if provided
            config.update(
                {
                    key: value
                    for key, value in {
                        "concepts_per_text": args.concepts_per_text,
                        "num_combinations": args.num_combinations,
                        "min_valid_cognates": args.min_valid_cognates,
                        "lexibank_path": args.lexibank_path,
                        "langs_per_entry": args.langs_per_entry,
                        "output_train_path": args.output_train_path,
                        "output_test_path": args.output_test_path,
                    }.items()
                    if value is not None
                }
            )

    # Display configuration being used
    print("Using the following configuration:")
    for key, value in config.items():
        if key == "test_folders":
            print(f"  test_folders: {len(value)} test folders")
        else:
            print(f"  {key}: {value}")

    # Create datasets with the final configuration
    create_datasets(
        lexibank_path=config["lexibank_path"],
        test_folders=config["test_folders"],
        num_combinations=config["num_combinations"],
        concepts_per_text=config["concepts_per_text"],
        min_valid_cognates=config["min_valid_cognates"],
        langs_per_entry=config["langs_per_entry"],
        output_train_path=config["output_train_path"],
        output_test_path=config["output_test_path"],
    )


if __name__ == "__main__":
    main()
