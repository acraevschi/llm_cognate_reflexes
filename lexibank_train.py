import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch import nn
from torch.utils.data import DataLoader
from train_helper import custom_dropout, shift_right, WeightedCrossEntropyLoss, run_epoch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
print(f"Using device: {device}")

folder = "torch_datasets"

train_data = torch.load(f"{folder}/train_dataset.pt")
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
print("Train data loaded")
del train_data

val_data = torch.load(f"{folder}/val_dataset.pt")
val_dataloader = DataLoader(val_data, batch_size=32, shuffle=True, generator=torch.Generator(device=device))
print("Validation data loaded")
del val_data

# Model and Tokenizer
model_name = "google/byt5-small"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.to(device)

# Optimizer and Scheduler
lr = 0.0001
optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
loss_fn = WeightedCrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1).to(device)

# Training Loop
n_epochs = 2

for n in range(n_epochs):
    print(f"Epoch {n + 1}/{n_epochs}")
    train_loss, val_loss = run_epoch(model, tokenizer, device, 
                                     train_dataloader, val_dataloader, 
                                     optim, loss_fn, scheduler=scheduler)
    print("_" * 100)