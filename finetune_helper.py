import random
import torch
from torch import nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def custom_dropout(
    input_ids_batch,
    tokenizer,
    dropout_prob=0.5,
    pad_prob=0.5,
    seq_drop_range=(0.1, 0.3),
):
    batch_size = input_ids_batch.size(0)
    selected_indices = random.sample(range(batch_size), int(batch_size * dropout_prob))

    for i in selected_indices:
        input_ids = input_ids_batch[i]
        input_length = (input_ids != tokenizer.pad_token_id).sum().item()
        dropout_fraction = random.uniform(seq_drop_range[0], seq_drop_range[1])
        num_dropout_tokens = int(input_length * dropout_fraction)
        non_pad_indices = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[
            0
        ]
        dropout_indices = random.sample(non_pad_indices.tolist(), num_dropout_tokens)

        for idx in dropout_indices:
            if random.random() < pad_prob:
                input_ids[idx] = tokenizer.pad_token_id
            else:
                input_ids[idx] = tokenizer.unk_token_id

    return input_ids_batch.to(device)


def shift_right(input_ids, pad_token_id):
    decoder_input_ids = input_ids.new_zeros(input_ids.shape)
    decoder_input_ids[:, 1:] = input_ids[:, :-1].clone()
    decoder_input_ids[:, 0] = pad_token_id
    return decoder_input_ids.to(device)


def run_epoch(
    model,
    tokenizer,
    device,
    train_data,
    val_data,
    optimizer,
    loss_fn,
    scheduler=None,
    augment=True,
):
    model.train()
    train_loss = 0
    train_bar = tqdm(train_data, desc="Training")

    for batch in train_bar:
        cognate_forms, attn_mask, labels, out_attn_mask = batch
        cognate_forms = cognate_forms.to(device)
        attn_mask = attn_mask.to(device)
        target_form = shift_right(labels.to(device), tokenizer.pad_token_id)
        out_attn_mask = out_attn_mask.to(device)

        if augment:
            cognate_forms = custom_dropout(cognate_forms, tokenizer)

        optimizer.zero_grad()
        outputs = model(
            input_ids=cognate_forms, attention_mask=attn_mask, labels=target_form
        )
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        train_bar.set_postfix(loss=train_loss / (train_bar.n + 1))

    model.eval()
    val_loss = 0
    val_bar = tqdm(val_data, desc="Validation")
    with torch.no_grad():
        for batch in val_bar:
            cognate_forms, attn_mask, labels, out_attn_mask = batch
            cognate_forms = cognate_forms.to(device)
            attn_mask = attn_mask.to(device)
            target_form = shift_right(labels.to(device), tokenizer.pad_token_id)
            out_attn_mask = out_attn_mask.to(device)

            outputs = model(
                input_ids=cognate_forms, attention_mask=attn_mask, labels=target_form
            )
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_loss += loss.item()
            val_bar.set_postfix(loss=val_loss / (val_bar.n + 1))

    train_loss = train_loss / len(train_data)
    val_loss = val_loss / len(val_data)

    if scheduler is not None:
        scheduler.step()

    return train_loss, val_loss
