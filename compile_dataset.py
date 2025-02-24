import os
import pandas as pd
import random
from datasets import Dataset
from tqdm import tqdm
import itertools

from lexibank_prep.lexibank_help import check_glotto_coverage
from glotto_trees.get_newick import get_phylogenetic_tree

random.seed(97)

test_folders = [
    "mannburmish",  # tonal, contains multiword forms, has a Glottocode, South-East Asia
    "gerarditupi",  # non-tonal, has a Glottocode, South/Centre of South America
    "savelyevturkic",  # non-tonal, vowel harmony, has a Glottocode, Eurasia
    "ratcliffearabic",  # non-tonal, fairly complex phonotactics, has a Glottocode, Middle East/Africa
    "walworthpolynesian",  # tonal, simple phonotactics occasional multimorphemic forms, has a Glottocode, Polynesia
]

# Global set to track test combinations across all folders
test_combinations = set()


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
    """Modified to work with subset of languages"""
    cognate_id = row["Cognate_ID"]

    # Get only the languages present in this chunk
    langs = row.index[1:]

    forms_original = [f"{lang} = {clean_form(row[lang])}" for lang in langs]
    forms_masked = [f"{lang} = {clean_form(masked_row[lang])}" for lang in langs]

    input_text = f"<{cognate_id}>\n" + "\n".join(forms_masked) + f"\n</{cognate_id}>\n"
    target_text = (
        f"<{cognate_id}>\n" + "\n".join(forms_original) + f"\n</{cognate_id}>\n"
    )

    return input_text, target_text


folders = os.listdir("lexibank")
dataset_entries = []
dataset_test_entries = []
concepts_per_text = 210  # maximal number of concepts per one input/output

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
    column_mapping = {
        langs["ID"][i]: (
            f"{langs['ID'][i]}:{langs['Glottocode'][i]}"
            if pd.notna(langs["Glottocode"][i])
            else langs["ID"][i]
        )
        for i in range(len(langs))
    }
    data.columns = [data.columns[0]] + [
        column_mapping.get(col, col) for col in data.columns[1:]
    ]

    # Get language columns and check if there are at least 3
    lang_columns = data.columns[1:].tolist()
    if len(lang_columns) < 3:
        continue  # Skip folders with fewer than 3 languages

    # Generate all possible 3-language combinations
    all_combinations = list(itertools.combinations(lang_columns, 3))
    random.shuffle(all_combinations)  # Randomize combination order

    # Process combinations based on dataset type
    if test_data:
        # For test folders: use first 100 combinations and register them
        selected_combinations = all_combinations[:100]
        # Add to global test combinations (sorted to avoid order variations)
        for combo in selected_combinations:
            test_combinations.add(tuple(sorted(combo)))
    else:
        # For training folders: filter out test combinations and take first 100 remaining
        filtered_combinations = [
            combo
            for combo in all_combinations
            if tuple(sorted(combo)) not in test_combinations
        ]
        selected_combinations = filtered_combinations[:100]

    # Process each selected combination
    for group in selected_combinations:
        group_columns = ["Cognate_ID"] + list(group)
        data_subset = data[group_columns]

        # Generate Newick tree (placeholder for actual implementation)
        newick = ""
        # try:
        #     glottocodes = [col.split(":")[1] if ":" in col else "" for col in group]
        #     newick = get_phylogenetic_tree(glottocodes)
        # except:
        #     newick = ""

        input_text = f"<NEWICK> {newick} </NEWICK>\n<Cognates>\n"
        target_text = "<Prediction>\n"

        # Shuffle and process concepts
        data_subset = data_subset.sample(frac=1).reset_index(drop=True)
        for i in range(min(len(data_subset), concepts_per_text)):
            row = data_subset.iloc[i]
            masked_row = mask_values(row)

            if len(masked_row) > 0:
                cog_input, cog_output = format_example(row, masked_row)
            else:
                cog_input, cog_output = format_example(row, row)

            input_text += cog_input
            target_text += cog_output

        input_text += "</Cognates>\n"
        target_text += "</Prediction>"

        # Add to appropriate dataset
        if test_data:
            dataset_test_entries.append({"input": input_text, "output": target_text})
        else:
            dataset_entries.append({"input": input_text, "output": target_text})

# Create final datasets
hf_dataset = Dataset.from_list(dataset_entries)
hf_test_dataset = Dataset.from_list(dataset_test_entries)

# # Save the dataset
hf_dataset.save_to_disk("hf_cognates_dataset", max_shard_size="50MB")
hf_test_dataset.save_to_disk("hf_cognates_test_dataset", max_shard_size="50MB")
print("Datasets saved successfully!")
