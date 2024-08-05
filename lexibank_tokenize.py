import random
import torch
from torch.utils.data import TensorDataset
from tokenize_help import split_cognates, get_non_empty_cognates
from transformers import AutoTokenizer
from tqdm import tqdm
import os



training = True # in case of False, surprise data is used
model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# def read_wide_format():
if training:
    test_dirs = [ # Training data
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

else:
    test_dirs = [ 
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

target_form = []
cognate_forms = []
lang_fam = []

val_target_form = []
val_cognate_forms = []
val_lang_fam = []

weight_count = dict()

lst_dirs = [folder for folder in os.listdir("lexibank") if os.path.isdir(os.path.join("lexibank", folder))]

random.seed(25)
for dir_name in tqdm(lst_dirs, desc="Tokenizing Lexibank"):
    if dir_name in test_dirs:
        continue
    file_path = f"lexibank/{dir_name}/wide_df.tsv"
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
    count = 0
    header = data[0]
    for i, head in enumerate(header):
        for n, data_point in enumerate(train_data):
        
            if (
                data_point[i] == ""
            ):  # dictionary keys are strings with target language and target form. If there is not target form, skip the iteration
                continue
            else:
                cognate_set = []
                for k in range(len(data_point)):
                    if k != i:
                        cognate_set.append(data_point[k])
                split_cognates_lst = split_cognates(cognate_set)

                for split_cognate in split_cognates_lst:
                    count += 1
                    cognate_forms.append(" | ".join(split_cognate))
                    target_form.append(data_point[i])         
                    lang_fam.append(dir_name)       
                
  
        for n, data_point in enumerate(val_data):
            if (
                data_point[i] == ""
            ):  # dictionary keys are strings with target language and target form. If there is not target form, skip the iteration
                continue
            else:
                val_cognate_langs = [header[k] for k in range(len(data_point)) if k != i]
                val_cognate_set = []
                for k in range(len(data_point)):
                    if k != i:
                        val_cognate_set.append(data_point[k])

                split_cognates_lst = split_cognates(val_cognate_set)

                for split_cognate in split_cognates_lst:
                    val_cognate_forms.append(" | ".join(split_cognate))
                    val_target_form.append(data_point[i])         
                    val_lang_fam.append(dir_name)
    weight_count[dir_name] = count

max_count = max(weight_count.values())
min_count = min(weight_count.values())
bin_size = (max_count - min_count) / 3

for key in weight_count:
    value = weight_count[key]
    if value >= max_count - bin_size:
        weight_count[key] = 0.75
    elif value >= max_count - 2 * bin_size:
        weight_count[key] = 1
    else:
        weight_count[key] = 1.25


fam_weight = list(map(lambda key: weight_count[key], lang_fam))
val_fam_weight = list(map(lambda key: weight_count[key], val_lang_fam))

# Tokenize input sequences
tokenized_inputs = tokenizer(cognate_forms, padding=True, truncation=True, return_tensors="pt")
# Tokenize output sequences
tokenized_outputs = tokenizer(target_form, padding=True, truncation=True, return_tensors="pt")
# Convert fam_weight to a tensor

fam_weight_tensor = torch.tensor(fam_weight).view(-1, 1)

for key in tokenized_inputs.keys():
    tokenized_inputs[key] = tokenized_inputs[key].to(dtype=torch.int8)
    tokenized_outputs[key] = tokenized_outputs[key].to(dtype=torch.int8)

# Create TensorDataset
dataset = TensorDataset(
    tokenized_inputs.input_ids,
    tokenized_inputs.attention_mask,
    tokenized_outputs.input_ids,
    tokenized_outputs.attention_mask,
    fam_weight_tensor
)

# Tokenize validation input sequences
val_tokenized_inputs = tokenizer(val_cognate_forms, padding=True, truncation=True, return_tensors="pt")

# Tokenize validation output sequences
val_tokenized_outputs = tokenizer(val_target_form, padding=True, truncation=True, return_tensors="pt")

# Convert val_fam_weight to a tensor
val_fam_weight_tensor = torch.tensor(val_fam_weight).view(-1, 1)

# Create validation TensorDataset
val_dataset = TensorDataset(
    val_tokenized_inputs.input_ids,
    val_tokenized_inputs.attention_mask,
    val_tokenized_outputs.input_ids,
    val_tokenized_outputs.attention_mask,
    val_fam_weight_tensor
)

# save the datasets
torch.save(dataset, "torch_datasets/train_dataset.pt")
torch.save(val_dataset, "torch_datasets/val_dataset.pt")