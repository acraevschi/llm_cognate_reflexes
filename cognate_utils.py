import Levenshtein
import pandas as pd
import random
import itertools

def clean_form(form):
    """Cleans the word form (removes split entries, handles missing data)."""
    if pd.isna(form) or form == "" or form == "-":
        return None
    form_lst = str(form).split()
    for i, f in enumerate(form_lst):
        # If multiple forms exist (e.g. "a/b"), take the first one
        if "/" in f:
            form_lst[i] = f.split("/")[0]
    return " ".join(form_lst)

def calculate_phonological_distance(row_a, row_b, context_langs):
    """
    Calculates normalized Levenshtein distance between two cognate sets 
    based ONLY on the context languages (excluding the target).
    """
    total_dist = 0
    valid_comps = 0
    
    for lang in context_langs:
        form_a = clean_form(row_a[lang])
        form_b = clean_form(row_b[lang])
        
        if form_a and form_b:
            # Normalized Levenshtein: 0 = identical, 1 = completely different
            dist = Levenshtein.distance(form_a, form_b)
            max_len = max(len(form_a), len(form_b))
            if max_len > 0:
                total_dist += dist / max_len
                valid_comps += 1
    
    # If they share no common languages with data, return max distance
    if valid_comps == 0:
        return 1.0
        
    return total_dist / valid_comps

def format_single_set(row, set_id, languages, target_lang=None):
    """Formats a single cognate set for the prompt."""
    lines = [f"<set_id={set_id}>"]
    for lang in languages:
        form = clean_form(row[lang])
        
        if lang == target_lang:
            val = "???"
        else:
            val = form if form else "-"
            
        lines.append(f"{lang}: {val}")
    lines.append(f"</set_id>")
    return "\n".join(lines)

def process_family_folder(folder_info):
    """
    Worker function to process a single language family folder.
    Strictly separates logic for Train Folders vs Test Folders.
    """
    (folder, lexibank_path, is_test_folder, num_combinations, 
     num_evidence_sets, min_valid_cognates, langs_per_entry, test_split_ratio) = folder_info

    local_train_entries = []
    local_test_entries = []

    data_path = f"{lexibank_path}/{folder}/wide_df.tsv"
    
    try:
        data = pd.read_csv(data_path, sep="\t", encoding="utf-8").fillna("-")
        lang_columns = data.columns[1:].tolist() # Skip Cognate_ID
        if len(lang_columns) < langs_per_entry:
            return [], []
    except Exception:
        return [], []

    # Generate language combinations
    all_combinations = list(itertools.combinations(lang_columns, langs_per_entry))
    random.seed(42) # Ensure reproducible shuffling
    random.shuffle(all_combinations)
    selected_combinations = all_combinations[:num_combinations]

    for combo in selected_combinations:
        combo_cols = ["Cognate_ID"] + list(combo)
        df_subset = data[combo_cols].copy()
        
        # Clean Data: Remove rows where ALL forms are missing
        mask = df_subset[list(combo)].apply(lambda x: x != "-").any(axis=1)
        df_subset = df_subset[mask]
        
        if len(df_subset) < min_valid_cognates:
            continue

        # Shuffle once
        df_subset = df_subset.sample(frac=1, random_state=42).reset_index(drop=True)

        # --- LOGIC BRANCHING ---
        if is_test_folder:
            # === TEST FOLDER MODE ===
            # Goal: Create Evaluation data only.
            # Strategy: Split folder into Reference (Evidence Pool) and Query (Test Set).
            # The model attempts to solve Query items using the Reference items.
            
            split_idx = int(len(df_subset) * (1 - test_split_ratio)) 
            
            evidence_pool_df = df_subset.iloc[:split_idx]
            query_rows_df = df_subset.iloc[split_idx:]
            
            target_list = local_test_entries # Save to TEST dataset
            
        else:
            # === TRAIN FOLDER MODE ===
            # Goal: Create Training data only.
            # Strategy: Every row is a valid training example.
            # For any specific row, the "Evidence Pool" is simply all OTHER rows in the folder.
            
            evidence_pool_df = df_subset
            query_rows_df = df_subset
            
            target_list = local_train_entries # Save to TRAIN dataset

        # --- PROCESS ROWS ---
        for _, query_row in query_rows_df.iterrows():
            # Iterate through each language in the combo to serve as the "Target"
            for target_lang in combo:
                target_form = clean_form(query_row[target_lang])
                
                # Skip if the target itself is missing (cannot verify reconstruction)
                if not target_form:
                    continue
                    
                # Context languages are everyone except the target
                context_langs = [l for l in combo if l != target_lang]
                
                # Check if query has at least one context form (otherwise impossible to predict)
                has_context = any(clean_form(query_row[l]) for l in context_langs)
                if not has_context:
                    continue

                # --- EVIDENCE SELECTION (Phonological N-Closest) ---
                distances = []
                for _, cand_row in evidence_pool_df.iterrows():
                    # CRITICAL: Do not use the query row itself as evidence
                    if cand_row["Cognate_ID"] == query_row["Cognate_ID"]:
                        continue
                        
                    dist = calculate_phonological_distance(query_row, cand_row, context_langs)
                    distances.append((dist, cand_row))
                
                # Sort by distance (asc) and take top N
                distances.sort(key=lambda x: x[0])
                top_evidence = [d[1] for d in distances[:num_evidence_sets]]
                
                if len(top_evidence) < num_evidence_sets:
                    continue

                # --- FORMATTING ---
                # 1. Build Evidence String
                evidence_str = ""
                for i, ev_row in enumerate(top_evidence):
                    evidence_str += format_single_set(ev_row, i+1, combo) + "\n"

                # 2. Build Query String (Target masked)
                query_str = format_single_set(query_row, len(top_evidence)+1, combo, target_lang)

                # 3. Construct Entry
                entry = {
                    "evidence": evidence_str,
                    "query": query_str,
                    "target_lang": target_lang,
                    "target_form": target_form,
                    "output": target_form
                }
                
                target_list.append(entry)

    return local_train_entries, local_test_entries
