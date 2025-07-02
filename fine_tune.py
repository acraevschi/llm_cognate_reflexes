from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from torch.cuda import empty_cache
import os
import torch
import gc
import json
from pathlib import Path

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

instruction_template = "### Input:\n"
response_template = "### Output:\n"

prompt = """
    You are a linguistic expert specializing in historical language reconstruction.

    # Task
    Analyze the list of cognates below and reconstruct any missing forms marked with "?".

    # Format
    - Input is provided as cognate sets in pseudo-XML format
    - Each concept is enclosed in its own tag
    - Forms marked with "?" need to be reconstructed
    - Forms marked with "-" represent unavailable data and should remain unchanged
    - Existing forms should not be modified

    # Instructions
    1. Examine patterns across the related language varieties
    2. Consider regular sound correspondences
    3. Reconstruct the missing forms based on linguistic principles
    4. Maintain the exact same structure in your output
    5. Wrap your complete response in <Reconstructed Cognates>...</Reconstructed Cognates> tags
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


def get_dataset_config_string(config):
    """Create a string representation of dataset config for checkpoint naming"""
    dataset_config = config["dataset"]
    return f"{dataset_config['concepts_per_text']}concepts_min{dataset_config['min_valid_cognates']}_{dataset_config['num_combinations']}combs"


def load_model_and_tokenizer(config):
    """Load model and tokenizer with configurations from config dict"""
    model_name = config["training"]["model_name"]
    max_length = config["training"]["max_length"]
    seed_num = config["training"]["seed_num"]

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
    except:
        print("Flash attention not working on this machine, trying without it...\n")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
    print("The model was successfully loaded!\n")

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed_num,
        max_seq_length=max_length,
        use_rslora=False,
        loftq_config=None,
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def formatting_prompts_func(example):
    return f"{prompt}\n{instruction_template}{example['input']}\n\n{response_template}{example['output']}"


def get_collator(tokenizer):
    return DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )


def prepare_datasets(config):
    """Load and prepare the dataset using config"""
    dataset_config = config["dataset"]
    dataset_string = get_dataset_config_string(config)
    train_data_path = f"{dataset_config['output_train_path']}/{dataset_string}"

    # Load and prepare the dataset
    data = load_from_disk(train_data_path)

    val_inds = data.num_rows // 10
    train_data = data.select(range(val_inds, data.num_rows))
    val_data = data.select(range(0, val_inds))

    # Format the datasets
    train_data = train_data.map(lambda ex: {"text": formatting_prompts_func(ex)})
    val_data = val_data.map(lambda ex: {"text": formatting_prompts_func(ex)})

    return train_data, val_data


def get_trainer(config, model, collator, train_dataset, eval_dataset):
    """Create the trainer with config parameters"""
    training_config = config["training"]
    model_name = training_config["model_name"].split("/")[1]
    dataset_string = get_dataset_config_string(config)
    checkpoint_path = (
        f"{training_config['checkpoint_dir']}/{model_name}/{dataset_string}"
    )

    training_args = SFTConfig(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        eval_accumulation_steps=2,
        optim="adamw_8bit",
        max_steps=training_config["total_steps"],
        save_steps=training_config["total_steps"] // 8,
        logging_steps=training_config["total_steps"] // 8,
        eval_strategy="steps",
        eval_steps=training_config["total_steps"] // 8,
        save_total_limit=4,
        warmup_ratio=0.1,
        learning_rate=training_config["learning_rate"],
        weight_decay=0.01,
        seed=training_config["seed_num"],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        max_seq_length=training_config["max_length"],
        bf16=is_bfloat16_supported(),
        dataset_num_proc=1,
        torch_compile=True,
        torch_empty_cache_steps=training_config["total_steps"] // 8 + 1,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        formatting_func=None,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
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

    # Make sure the checkpoint directory exists
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Clean memory before training
    empty_cache()
    gc.collect()

    model, tokenizer = load_model_and_tokenizer(config)
    train_dataset, eval_dataset = prepare_datasets(config)
    collator = get_collator(tokenizer)
    trainer, checkpoint_path = get_trainer(
        config, model, collator, train_dataset, eval_dataset
    )

    print("\n\n---Starting training---\n\n")
    trainer.train()
    print("\n\n---Training complete!---\n\n")

    path_best_model = checkpoint_path + "/best_model"
    best_model_path = trainer.save_model(path_best_model)
    tokenizer.save_pretrained(path_best_model)

    empty_cache()
    gc.collect()

    print(f"Best model saved at: {path_best_model}")
    return path_best_model


if __name__ == "__main__":
    train_model()
