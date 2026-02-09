import Levenshtein
import pandas as pd
import numpy as np
import random
import itertools


def clean_form(form):
    """Cleans the word form (removes split entries, handles missing data)."""
    if pd.isna(form) or form == "" or form == "-":
        return None
    # Convert to string to be safe
    form = str(form)
    # If multiple forms exist (e.g. "a/b"), take the first one
    if "/" in form:
        form = form.split("/")[0]
    # Remove extra whitespace
    return " ".join(form.split())


def calc_normalized_levenshtein(target_str, candidate_list):
    """
    Helper to calculate normalized Levenshtein distance between a target string
    and a list of candidate strings.
    Returns a numpy array of distances (0.0 to 1.0).
    """
    if not target_str:
        # If target is missing, distance is max (1.0) for all candidates
        return np.ones(len(candidate_list), dtype=float)

    dists = []
    for cand in candidate_list:
        if cand is None:
            dists.append(1.0)
        else:
            d = Levenshtein.distance(target_str, cand)
            max_len = max(len(target_str), len(cand))
            if max_len == 0:
                dists.append(1.0)
            else:
                dists.append(d / max_len)

    return np.array(dists, dtype=float)


def format_single_set(row_dict, set_id, languages, lang_mapping, target_lang=None):
    """
    Formats a single cognate set for the prompt.
    Uses the pre-calculated lang_mapping to determine labels (A, B, C...).
    """
    lines = [f"<{set_id}>"]

    for lang in languages:
        # Retrieve the specific letter code for this language
        lang_code = lang_mapping[lang]
        val = row_dict.get(lang)

        if lang == target_lang:
            display_val = "???"
        else:
            display_val = val if val else "-"

        lines.append(f"{lang_code}: {display_val}")
    return "\n".join(lines)


def process_family_folder(folder_info):
    """
    Worker function to process a single language family folder.
    Optimized with Vectorized Distance Calculation and Evidence Density Filtering.
    """
    (
        folder,
        lexibank_path,
        is_test_folder,
        num_combinations,
        num_evidence_sets,
        min_valid_cognates,
        langs_per_entry,
        test_split_ratio,
    ) = folder_info

    local_train_entries = []
    local_test_entries = []

    data_path = f"{lexibank_path}/{folder}/wide_df.tsv"

    try:
        data = pd.read_csv(data_path, sep="\t", encoding="utf-8").fillna("-")
        lang_columns = data.columns[1:].tolist()  # Skip Cognate_ID
        if len(lang_columns) < langs_per_entry:
            return [], []
    except Exception:
        return [], []

    # Generate language combinations
    all_combinations = list(itertools.combinations(lang_columns, langs_per_entry))
    random.seed(97)
    random.shuffle(all_combinations)
    selected_combinations = all_combinations[:num_combinations]

    for combo in selected_combinations:
        # Create Language Mapping for this combination (A, B, C...)
        lang_letter_map = {lang: chr(65 + i) for i, lang in enumerate(combo)}

        combo_cols = ["Cognate_ID"] + list(combo)
        df_subset = data[combo_cols].copy()

        # 1. OPTIMIZATION: Pre-clean the entire dataframe at once
        for col in combo:
            df_subset[col] = df_subset[col].apply(clean_form)

        # Remove rows where ALL forms are missing (after cleaning)
        mask = df_subset[list(combo)].notna().any(axis=1)
        df_subset = df_subset[mask]

        if len(df_subset) < min_valid_cognates:
            continue

        # Shuffle once
        df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- LOGIC BRANCHING ---
        if is_test_folder:
            split_idx = int(len(df_subset) * (1 - test_split_ratio))
            evidence_pool_df = df_subset.iloc[:split_idx]
            query_rows_df = df_subset.iloc[split_idx:]
            target_list = local_test_entries
        else:
            evidence_pool_df = df_subset
            query_rows_df = df_subset
            target_list = local_train_entries

        # 2. OPTIMIZATION: Convert evidence pool to Dictionary of Numpy Arrays
        evidence_ids = evidence_pool_df["Cognate_ID"].to_numpy()
        evidence_dict = {col: evidence_pool_df[col].to_numpy() for col in combo}
        evidence_records = evidence_pool_df.to_dict("records")

        # --- PROCESS ROWS ---
        for _, query_row in query_rows_df.iterrows():

            for target_lang in combo:
                target_form = query_row[target_lang]

                if not target_form:
                    continue

                context_langs = [l for l in combo if l != target_lang]

                # Check if query itself has at least one context form
                if not any(query_row[l] for l in context_langs):
                    continue

                # --- 3. OPTIMIZATION: Vectorized Evidence Selection ---
                num_candidates = len(evidence_ids)
                total_dist_vector = np.zeros(num_candidates, dtype=float)
                valid_counts_vector = np.zeros(num_candidates, dtype=float)

                for ctx_lang in context_langs:
                    query_val = query_row[ctx_lang]
                    candidate_vals = evidence_dict[ctx_lang]

                    dists = calc_normalized_levenshtein(query_val, candidate_vals)

                    is_valid_evidence = candidate_vals != None

                    if query_val:
                        total_dist_vector += dists
                        valid_counts_vector += is_valid_evidence.astype(float)

                no_valid_comps_mask = valid_counts_vector == 0
                valid_counts_vector[no_valid_comps_mask] = 1.0

                avg_distances = total_dist_vector / valid_counts_vector
                avg_distances[no_valid_comps_mask] = 1.0

                self_match_mask = evidence_ids == query_row["Cognate_ID"]
                avg_distances[self_match_mask] = 999.0

                # --- FIND TOP N ---
                if len(avg_distances) < num_evidence_sets:
                    continue

                top_indices = np.argpartition(avg_distances, num_evidence_sets)[
                    :num_evidence_sets
                ]
                top_indices = top_indices[np.argsort(avg_distances[top_indices])]
                top_evidence = [evidence_records[i] for i in top_indices]

                # --- 4. NEW: EVIDENCE DENSITY FILTER ---
                # "Get rid of cases where evidence is useless"
                # Check if every context language is present in at least 50% of the evidence rows.
                is_evidence_usable = True
                for ctx_lang in context_langs:
                    # Count rows where this language is NOT None
                    valid_count = sum(
                        1 for row in top_evidence if row.get(ctx_lang) is not None
                    )
                    if valid_count < (len(top_evidence) * 0.5):
                        is_evidence_usable = False
                        break

                if not is_evidence_usable:
                    continue

                # --- FORMATTING ---
                evidence_str = ""
                for i, ev_row in enumerate(top_evidence):
                    evidence_str += (
                        format_single_set(ev_row, i + 1, combo, lang_letter_map) + "\n"
                    )

                query_str = format_single_set(
                    query_row.to_dict(),
                    len(top_evidence) + 1,
                    combo,
                    lang_letter_map,
                    target_lang,
                )

                entry = {
                    "evidence": evidence_str,
                    "query": query_str,
                    "target_lang": lang_letter_map[target_lang],  # Encoded "C"
                    "lang_map": lang_letter_map,  # Full map
                    "target_form": target_form,
                    "output": target_form,
                    "family": folder,
                }

                target_list.append(entry)

    return local_train_entries, local_test_entries
