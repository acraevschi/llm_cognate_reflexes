import os
import json
import gc
import torch
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import bitsandbytes
from torch.cuda import empty_cache
from metrics import calculate_ned

# Template for the Encoder Input
prompt_base = """Reconstruct cognates: 
{evidence}
# Target:
{query}
"""


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing config file: {e}")
        return None


def load_model_and_tokenizer(config):
    """Load ByT5 model and tokenizer for full fine-tuning"""
    model_name = config["model_name"]

    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model for Seq2Seq. ByT5 works best with bfloat16 if hardware supports it.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    )

    print("The model was successfully loaded!\n")
    return model, tokenizer


def preprocess_function(
    examples, tokenizer, max_input_length=1280, max_target_length=64
):
    """
    Tokenize inputs (Prompt) and targets (Ground Truth Word) separately
    for Encoder-Decoder architecture.
    """
    inputs = []
    targets = []

    for i in range(len(examples["evidence"])):
        # Construct the input prompt
        input_text = prompt_base.format(
            evidence=examples["evidence"][i],
            query=examples["query"][i],
        )
        inputs.append(input_text)

        # The target is just the output word
        targets.append(examples["target_form"][i])

    # Tokenize inputs
    model_inputs = tokenizer(
        inputs,
        max_length=max_input_length,
        truncation=True,
        padding=False,  # Padding handled by collator
    )

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, max_length=max_target_length, truncation=True, padding=False
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """
    Compute NED metric using generated predictions.
    This replaces the GenerativeEvalCallback.
    """
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    # Decode generated predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them (these are ignored indices)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    ned_scores = []
    for pred, label in zip(decoded_preds, decoded_labels):
        ned = calculate_ned(pred, label)
        ned_scores.append(ned)

    avg_ned = np.mean(ned_scores) if ned_scores else 0.0

    return {"eval_ned": avg_ned}


def prepare_datasets(config, tokenizer):
    """Load and process the dataset"""
    dataset_config = config["dataset"]
    dataset_string = f"{dataset_config['langs_per_entry']}langs_{dataset_config['num_evidence_sets']}evidence"
    train_data_path = f"{dataset_config['output_train_path']}/{dataset_string}"

    print(f"Loading dataset from: {train_data_path}")
    data = load_from_disk(train_data_path)

    # Shuffle
    data = data.shuffle(seed=42)

    # Split
    eval_percent = config["training"].get("eval_percent", 10)
    eval_inds = data.num_rows // eval_percent
    train_data = data.select(range(eval_inds, data.num_rows))
    eval_data = data.select(range(0, eval_inds))

    # Preprocess (Tokenize)
    print("Tokenizing datasets...")
    fn_kwargs = {
        "tokenizer": tokenizer,
        "max_input_length": config["training"].get("max_length", 512),
        "max_target_length": 64,  # Cognates are short
    }

    train_data = train_data.map(
        preprocess_function,
        batched=True,
        fn_kwargs=fn_kwargs,
        remove_columns=data.column_names,  # Remove raw text columns to save memory
    )
    eval_data = eval_data.map(
        preprocess_function,
        batched=True,
        fn_kwargs=fn_kwargs,
        remove_columns=data.column_names,
    )

    return train_data, eval_data


def get_trainer(config, model, tokenizer, train_dataset, eval_dataset):
    """Create the Seq2SeqTrainer"""
    training_config = config["training"]
    model_name_short = config["model_name"].split("/")[-1]

    checkpoint_path = f"{training_config['checkpoint_dir']}/{model_name_short}/"

    # Versioning logic
    if not os.path.exists(checkpoint_path):
        checkpoint_path += "run_0/"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    else:
        existing_runs = [
            int(folder.name.split("_")[-1])
            for folder in Path(checkpoint_path).iterdir()
            if folder.is_dir() and folder.name.startswith("run_")
        ]
        if existing_runs:
            last_run = max(existing_runs)
            checkpoint_path += f"run_{last_run + 1}/"
        else:
            checkpoint_path += "run_0/"
            Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    num_of_evals = training_config.get("num_of_evals", 10)

    # Seq2Seq specific arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        # Batch sizes
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        # Training loop
        max_steps=training_config["total_steps"],
        save_steps=training_config["total_steps"] // num_of_evals,
        logging_steps=training_config["total_steps"] // num_of_evals,
        eval_strategy="steps",
        eval_steps=training_config["total_steps"] // num_of_evals,
        # Optimization
        optim="adamw_8bit",
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_ratio=training_config["warmup_ratio"],
        weight_decay=0.1,
        # Checkpointing
        save_total_limit=5,
        evaluation_strategy="steps",
        eval_accumulation_steps=1
        load_best_model_at_end=True,
        metric_for_best_model="eval_ned",  # We want to minimize NED usually, but if metric is similarity, maximize
        greater_is_better=False,  # NED is distance, lower is better
        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        # Generation for metrics
        predict_with_generate=True,
        generation_max_length=64,  # Max length for the output word
        # Misc
        seed=training_config["seed_num"],
        dataloader_num_workers=4,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True
    )

    # Wrap compute_metrics to pass tokenizer
    def compute_metrics_wrapper(eval_preds):
        return compute_metrics(eval_preds, tokenizer)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_wrapper,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=training_config["early_stopping_threshold"],
            )
        ],
    )

    return trainer, checkpoint_path


def train_model(config_path="config.json"):
    """Main function to train model using config file"""
    config = load_config(config_path)
    if not config:
        return

    # Clean memory
    empty_cache()
    gc.collect()

    # Load Model & Tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Prepare Data
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # Get Trainer
    trainer, checkpoint_path = get_trainer(
        config, model, tokenizer, train_dataset, eval_dataset
    )

    # Write config copy
    with open(checkpoint_path + "/config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n\n---Starting training---\n\n")
    trainer.train()
    print("\n\n---Training complete!---\n\n")

    path_best_model = checkpoint_path + "/best_model"
    trainer.save_model(path_best_model)
    tokenizer.save_pretrained(path_best_model)

    empty_cache()
    gc.collect()

    print(f"Best model saved at: {path_best_model}")
    return path_best_model


if __name__ == "__main__":
    train_model()