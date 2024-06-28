import random
import torch
from torch.utils.data import TensorDataset
from transformers import AutoTokenizer
from tokenize_help import split_cognates, get_non_empty_cognates
import json

proportion = 0.5
assert proportion in [0.10, 0.20, 0.30, 0.40, 0.50]
prop_path = str(proportion) + "0" # to match the string in the file name

model_name = "google/byt5-small"

training = True # in case of False, surprise data is used

if training:
    lst_dirs = [ # Training data
        "abrahammonpa",
        "allenbai",
        "backstromnorthernpakistan",
        "castrosui",
        "davletshinaztecan",
        "felekesemitic",
        "hantganbangime",
        "hattorijaponic",
        "listsamplesize",
        "mannburmish",
    ]
    main_dir = "training"

else:
    lst_dirs = [ 
        "bantubvd",
        "beidazihui",
        "birchallchapacuran",
        "bodtkhobwa",
        "bremerberta",
        "deepadungpalaung",
        "hillburmish",
        "kesslersignificance",
        "luangthongkumkaren",
        "wangbai"
    ]
    main_dir = "surprise"


tokenizer = AutoTokenizer.from_pretrained(model_name)
lang_token_mapping = {}

target_form = []
cognate_forms = []

val_target_form = []
val_cognate_forms = []

num_langs = set()

random.seed(25)
for dir_name in lst_dirs:
    file_path = "ST2022/data/" + dir_name + f"/training-{prop_path}.tsv" if training \
        else "ST2022/data-surprise/" + dir_name + f"/training-{prop_path}.tsv"
    with open(
        file_path,
        encoding="UTF-8",
    ) as f:
        file = f.readlines()
        data = list(map(lambda x: x.strip("\n").split("\t")[1:], file))
        
        val_num = int(len(data) * 0.1)
        
        start_index = random.randint(1, len(data)-val_num)
        end_index = start_index + val_num

        train_data = data[1:start_index] + data[end_index:]

        val_data = data[start_index:end_index]        

    header = data[0]
    for i, head in enumerate(header):
        lang_token_mapping[head] = f"<extra_id_{i}>"
        additional_special_tokens = list(lang_token_mapping.values())
        tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens})

    # Save lang_token_mapping as json
    with open(file_path.rsplit('/', 1)[0] + '/lang_token_mapping.json', 'w') as f:
        json.dump(lang_token_mapping, f)

    # Save the modified tokenizer
    tokenizer.save_pretrained(file_path.rsplit('/', 1)[0] + "/adapted_tokenizer")

    for i, head in enumerate(header):
        for n, data_point in enumerate(train_data):
            if (
                data_point[i] == ""
            ):  # dictionary keys are strings with target language and target form. If there is not target form, skip the iteration
                continue
            else:
                cognate_set = [data_point[k] for k in range(len(data_point)) if k != i]
                cognate_langs = [header[k] for k in range(len(data_point)) if k != i]

                split_cognates_lst = split_cognates(cognate_set, cognate_langs, lang_token_mapping)

                for split_cognate in split_cognates_lst:
                    cognate_forms.append(" | ".join(split_cognate) + f" | {lang_token_mapping[head]}") # add target language to cognate forms
                    target_form.append(data_point[i])         
                                    
        for n, data_point in enumerate(val_data):
            if (
                data_point[i] == ""
            ):  # dictionary keys are strings with target language and target form. If there is not target form, skip the iteration
                continue
            else:
                cognate_set = [data_point[k] for k in range(len(data_point)) if k != i]
                cognate_langs = [header[k] for k in range(len(data_point)) if k != i]

                split_cognates_lst = split_cognates(cognate_set, cognate_langs, lang_token_mapping)

                for split_cognate in split_cognates_lst:
                    val_cognate_forms.append(" | ".join(split_cognate) + f" | {lang_token_mapping[head]}") # add target language to cognate forms
                    val_target_form.append(data_point[i])
        
    # Tokenize input sequences
    tokenized_inputs = tokenizer(cognate_forms, padding=True, truncation=True, return_tensors="pt")
    # Tokenize output sequences
    tokenized_outputs = tokenizer(target_form, padding=True, truncation=True, return_tensors="pt")
        
    # Create TensorDataset
    dataset = TensorDataset(
        tokenized_inputs.input_ids,
        tokenized_inputs.attention_mask,
        tokenized_outputs.input_ids,
        tokenized_outputs.attention_mask
    )

    # Tokenize validation input sequences
    val_tokenized_inputs = tokenizer(val_cognate_forms, padding=True, truncation=True, return_tensors="pt")

    # Tokenize validation output sequences
    val_tokenized_outputs = tokenizer(val_target_form, padding=True, truncation=True, return_tensors="pt")

    # Create validation TensorDataset
    val_dataset = TensorDataset(
        val_tokenized_inputs.input_ids,
        val_tokenized_inputs.attention_mask,
        val_tokenized_outputs.input_ids,
        val_tokenized_outputs.attention_mask
    )

    # save the datasets
    file_path.rsplit('/', 1)[0]
    torch.save(dataset, f"{file_path.rsplit('/', 1)[0]}/train_dataset_{prop_path}.pt")
    torch.save(val_dataset, f"{file_path.rsplit('/', 1)[0]}/val_dataset_{prop_path}.pt")
