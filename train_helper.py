import random
import torch
from torch import nn
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

def custom_dropout(input_ids_batch, tokenizer, dropout_prob=0.5, pad_prob=0.5, seq_drop_range=(0.1, 0.3)):
    batch_size = input_ids_batch.size(0)
    selected_indices = random.sample(range(batch_size), int(batch_size * dropout_prob))

    for i in selected_indices:
        input_ids = input_ids_batch[i]
        input_length = (input_ids != tokenizer.pad_token_id).sum().item()
        dropout_fraction = random.uniform(seq_drop_range[0], seq_drop_range[1])
        num_dropout_tokens = int(input_length * dropout_fraction)
        non_pad_indices = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
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

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index, label_smoothing=0.1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, logits, targets, weights):
        # Ensure targets are on the same device as logits
        targets = targets.to(logits.device)
        
        log_probs = self.log_softmax(logits)
        targets = targets.view(-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        num_classes = logits.size(-1)
        one_hot_targets = torch.zeros_like(log_probs).scatter(1, targets.unsqueeze(1), 1)
        smoothed_targets = (1 - self.label_smoothing) * one_hot_targets + self.label_smoothing / num_classes
        
        weights = torch.where(targets == self.ignore_index, torch.tensor(0.0, device=weights.device), weights)
        nll_loss = nll_loss * weights

        loss = nll_loss.mean()
        return loss

def evaluate_model(model, tokenizer, val_data, loss_fn):
    model.eval()
    val_loss = 0
    val_bar = tqdm(val_data, desc="Validation")
    with torch.no_grad():
        for batch in val_bar:
            cognate_forms, attn_mask, labels, out_attn_mask, weight = batch
            cognate_forms = cognate_forms.to(device)
            attn_mask = attn_mask.to(device)
            target_form = shift_right(labels.to(device), tokenizer.pad_token_id)
            out_attn_mask = out_attn_mask.to(device)
            weight = weight.to(device)

            outputs = model(input_ids=cognate_forms, attention_mask=attn_mask, labels=target_form)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1), weight)
            val_loss += loss.item()
            val_bar.set_postfix(loss=val_loss/(val_bar.n + 1))
    val_loss = val_loss / len(val_data)
    
    return val_loss

def run_epoch(model, tokenizer, device, train_data, val_data, optimizer, loss_fn, scheduler=None, augment=True, evals_per_epoch=4):
    steps_to_eval = len(train_data) // evals_per_epoch
    eval_points = [steps_to_eval*i for i in range(1, evals_per_epoch)] + [len(train_data)]

    val_loss_lst = []
    stop_training = False

    model.train()
    train_loss = 0
    train_bar = tqdm(train_data, desc="Training")

    for batch in train_bar:
        if stop_training:
            print("Validation loss has not decreased, stop training")
            break
        cognate_forms, attn_mask, labels, out_attn_mask, weight = batch
        cognate_forms = cognate_forms.to(device)
        attn_mask = attn_mask.to(device)
        target_form = shift_right(labels.to(device), tokenizer.pad_token_id)
        out_attn_mask = out_attn_mask.to(device)
        weight = weight.to(device)

        if augment:
            cognate_forms = custom_dropout(cognate_forms, tokenizer)
        
        optimizer.zero_grad()
        outputs = model(input_ids=cognate_forms, attention_mask=attn_mask, labels=target_form)
        logits = outputs.logits
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1), weight)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        current_step = train_bar.n + 1
        train_bar.set_postfix(loss=train_loss/current_step)

        if current_step in eval_points:
            val_loss = evaluate_model(model, tokenizer, val_data, loss_fn)
            val_loss_lst.append(val_loss)
            if len(val_loss_lst) > 1 and val_loss_lst[-1] > val_loss_lst[-2]:
                stop_training = True
            else:
                torch.save(model.state_dict(), "checkpoint_byt5_cognates.pt")

    train_loss = train_loss / current_step

    if scheduler is not None:
        scheduler.step()

    return train_loss, val_loss