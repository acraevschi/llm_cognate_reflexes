import random 

def get_non_empty_cognates(cognate_set):
    non_empty_inds = [i for i, x in enumerate(cognate_set) if x != ""]
    return non_empty_inds

def split_cognates(cognate_set, cognate_langs=None, lang_token_mapping=None, max_num_cognates=10):
    non_empty_inds = get_non_empty_cognates(cognate_set)
    num_non_empty = len(non_empty_inds)
    random.seed(25)
    random.shuffle(non_empty_inds)
    if cognate_langs != None and lang_token_mapping != None:
        cognate_forms = [f"{lang_token_mapping[cognate_langs[i]]} {cognate_set[i]}" for i in non_empty_inds]
    else:
        cognate_forms = [cognate_set[i] for i in non_empty_inds]

    if num_non_empty <= max_num_cognates:
        return [cognate_forms] # need to embed in list to allow for iteration
    else:
        split_cognates_lst = []
        num_splits = num_non_empty // max_num_cognates
        cognates_per_split = num_non_empty // num_splits
        for i in range(num_splits):
            start = i * cognates_per_split
            end = (i + 1) * cognates_per_split
            split_cognates_lst.append(cognate_forms[start:end])
        if num_non_empty % cognates_per_split != 0:
            split_cognates_lst.append(cognate_forms[num_splits * cognates_per_split:])
        
        return split_cognates_lst
