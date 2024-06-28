import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from train_helper import run_epoch
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

############### Make proportion and training arguments ################
proportion = 0.5
assert proportion in [0.10, 0.20, 0.30, 0.40, 0.50]
prop_path = str(proportion) + "0" # to match the string in the file name

training = True # in case of False, surprise data is used


model_name = "google/byt5-small"


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

for dir_name in lst_dirs:
    file_path = "ST2022/data/" + dir_name if training else "ST2022/data-surprise/" + dir_name
    tokenizer = AutoTokenizer.from_pretrained(file_path + "/adapted_tokenizer")
    lang_token_mapping = json.load(open(file_path + "/lang_token_mapping.json"))

        
    train_data = torch.load(f"{file_path}/train_dataset_{prop_path}.pt")
    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
    print("Train data loaded")
    del train_data

    val_data = torch.load(f"{file_path}/val_dataset_{prop_path}.pt")
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
    print("Validation data loaded")
    del val_data


    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.load_state_dict(torch.load("checkpoint.pt"))
    model.to(device)

    
    # Optimizer and Scheduler
    lr = 0.00001
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1).to(device)
        
    # Training Loop
    n_epochs = 15
    train_loss_lst = []
    val_loss_lst = []
    patience = 3
    patience_counter = 0

    for n in range(n_epochs):
        print(f"Epoch {n + 1}/{n_epochs}")
        train_loss, val_loss = run_epoch(model, tokenizer, device, 
                                        train_dataloader, val_dataloader, 
                                        optim, loss_fn, scheduler=scheduler, 
                                        finetune=True)
        train_loss_lst.append(train_loss)
        val_loss_lst.append(val_loss)
        if n > 0:
            if val_loss_lst[-1] > min(val_loss_lst[:-1]):
                print("Validation loss has not decreased")
                patience_counter += 1
                if patience_counter == patience:
                    print("Early stopping")
                    break
            else:
                print("Validation loss has decreased")
                torch.save(model.state_dict(), f"{file_path}/checkpoint_{dir_name}_{prop_path}.pt")
                print(f"Model saved to {file_path}/checkpoint_{dir_name}_{prop_path}.pt")
        print("_" * 100)