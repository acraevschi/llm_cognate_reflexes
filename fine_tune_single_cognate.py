from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from torch.cuda import empty_cache
import os
import torch
import gc
import json
from pathlib import Path
import numpy as np

from hf_token import HF_TOKEN


prompt_base = """Comparative Linguistics Reconstruction Data

== Context ==
Evidence Data:
{evidence}

== Task Configuration ==
Target Language: {target_lang}

== Input Query ==
{query}

== Reconstructed Form ==
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
    """Load model and tokenizer with configurations from config dict using Standard Transformers + Peft"""
    model_name = config["training"]["model_name"]
    max_length = config["training"]["max_length"]
    seed_num = config["training"]["seed_num"]
    r = config["training"]["r"]
    lora_alpha = config["training"]["lora_alpha"]

    # Determine dtype support
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading Tokenizer for {model_name}...")
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_length,
        padding_side="right",
        use_fast=False,
        token=HF_TOKEN # Uses the token logged in via huggingface_hub
    )
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Model {model_name}...")
    # Load Model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
            token=HF_TOKEN # Uses the token logged in via huggingface_hub
        )
    except Exception as e:
        print(f"Flash attention not working or error loading: {e}. Trying fallback...\n")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            token=HF_TOKEN
            )

    print("The model was successfully loaded!\n")

    # Configure LoRA using PEFT
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA adapter
    model = get_peft_model(model, peft_config)

    # Enable gradient checkpointing manually
    # model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    return model, tokenizer


def formatting_prompts_func(example, eos_token):
    """
    Constructs the dictionary with 'prompt' and 'completion' keys.
    """
    # 1. Build the Prompt
    input_text = prompt_base.format(
        evidence=example['evidence'],
        query=example['query'],
        target_lang=example['target_lang']
    )



    # 2. Get the Completion (Output + EOS)
    completion_text = f"{example['output']}"

    return {
        "prompt": input_text,
        "completion": completion_text,
    }


def prepare_datasets(config, eos_token, eval_percent=10):
    """Load and prepare the dataset into prompt/completion columns"""
    dataset_config = config["dataset"]
    dataset_string = f"{dataset_config['langs_per_entry']}langs_{dataset_config['num_evidence_sets']}evidence"
    train_data_path = f"{dataset_config['output_train_path']}/{dataset_string}"

    print(f"Loading dataset from: {train_data_path}")
    data = load_from_disk(train_data_path)

    data = data.shuffle(seed=42)

    # Map to new structure and remove old columns
    original_columns = data.column_names
    data = data.map(
        lambda ex: formatting_prompts_func(ex, eos_token),
        remove_columns=original_columns
    )

    # Split train/eval
    eval_inds = data.num_rows // eval_percent
    train_data = data.select(range(eval_inds, data.num_rows))
    eval_data = data.select(range(0, eval_inds))

    return train_data, eval_data


def get_trainer(config, model, tokenizer, train_dataset, eval_dataset):
    """Create the trainer with config parameters"""
    training_config = config["training"]
    model_name = training_config["model_name"].split("/")[1]

    checkpoint_path = (
        f"{training_config['checkpoint_dir']}/{model_name}/"
    )

    if not os.path.exists(checkpoint_path):
        checkpoint_path += "run_0/"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    else:
        existing_runs = [
            int(folder.name.split("_")[-1]) for folder in Path(checkpoint_path).iterdir() if folder.is_dir() and folder.name.startswith("run_")
        ]
        if existing_runs:
            last_run = max(existing_runs)
            checkpoint_path += f"run_{last_run + 1}/"

    num_of_evals = config["training"].get("num_of_evals", 10)

    # Use SFTConfig
    training_args = SFTConfig(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        completion_only_loss=True,
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        eval_accumulation_steps=2,
        optim="adamw_8bit",
        max_steps=training_config["total_steps"],
        save_steps=training_config["total_steps"] // num_of_evals,
        logging_steps=training_config["total_steps"] // num_of_evals,
        eval_strategy="steps",
        eval_steps=training_config["total_steps"] // num_of_evals,
        save_total_limit=5,
        warmup_ratio=training_config["warmup_ratio"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        max_length=training_config["max_length"],
        weight_decay=0.01,
        seed=training_config["seed_num"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        dataset_num_proc=4,
        torch_compile=False,
        report_to="none",
    )


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=training_config["early_stopping_threshold"],
            ),
        ],
    )

    return trainer, checkpoint_path


def train_model(config_path="config.json"):
    """Main function to train model using config file"""
    config = load_config(config_path)
    if not config:
        return

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    empty_cache()
    gc.collect()

    model, tokenizer = load_model_and_tokenizer(config)
    eos_token = tokenizer.eos_token

    # Prepare data
    train_dataset, eval_dataset = prepare_datasets(config, eos_token, eval_percent=config["training"]["eval_percent"])

    # Get trainer
    trainer, checkpoint_path = get_trainer(
        config, model, tokenizer, train_dataset, eval_dataset
    )

    config_path = checkpoint_path + "/config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print("\n\n---Starting training---\n\n")
    trainer.train()
    print("\n\n---Training complete!---\n\n")

    path_best_model = checkpoint_path + "/best_model"
    # Save adapter
    trainer.save_model(path_best_model)
    tokenizer.save_pretrained(path_best_model)

    empty_cache()
    gc.collect()

    print(f"Best model saved at: {path_best_model}")
    return path_best_model

if __name__ == "__main__":
    train_model()
