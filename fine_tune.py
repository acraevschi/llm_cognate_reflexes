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

instruction_template = "\n### Instruction:\n"
input_template = "### Input:\n"
response_template = "### Output:\n"

prompt = """
### Instruction:
You are a linguistic expert specializing in historical language reconstruction.

# Task
Analyze the cognate sets provided and reconstruct any missing word forms marked with "?".

# Input Format
- Multiple cognate sets are provided within <Cognates>...</Cognates> tags
- Each cognate set has its own ID tag <cognate_id>...</cognate_id>
- Each language form is listed as: language_name = word_form
- All phonological segments within a word_form are separated by spaces
- Forms marked with "?" need to be reconstructed
- Forms marked with "-" represent unavailable data
- If a a cognate consists of multiple words, they will be separated by a "+" sign

# Output Format
- Your response should ONLY include the reconstructed forms (marked with "?" in input)
- Maintain the same tag structure with <cognate_id>...</cognate_id> for each set
- Include ONLY the previously masked languages that need reconstruction
- Wrap your complete response in <Reconstructed Cognates>...</Reconstructed Cognates> tags

# Reconstruction Guidelines
1. Identify systematic sound correspondences across language varieties
2. Consider phonological patterns and regular sound changes
3. Use comparative method principles to infer the missing forms
4. Ensure reconstructed forms follow plausible phonotactic patterns
5. Do not modify any existing forms or include them in your output
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
    r = config["training"]["r"]
    lora_alpha = config["training"]["lora_alpha"]

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
        r=r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
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


def formatting_prompts_func(example, eos_token):
    return f"{prompt}\n{input_template}{example['input']}\n\n{response_template}{example['output']}{eos_token}"


def get_collator(tokenizer):
    return DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )


def prepare_datasets(config, eos_token):
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
    train_data = train_data.map(lambda ex: {"text": formatting_prompts_func(ex, eos_token)})
    val_data = val_data.map(lambda ex: {"text": formatting_prompts_func(ex, eos_token)})

    return train_data, val_data


def get_trainer(config, model, tokenizer, collator, train_dataset, eval_dataset):
    """Create the trainer with config parameters"""
    training_config = config["training"]
    model_name = training_config["model_name"].split("/")[1]
    
    checkpoint_path = (
        f"{training_config['checkpoint_dir']}/{model_name}/"
    )

    # check if there are folders in the checkpoint path
    if not os.path.exists(checkpoint_path):
        checkpoint_path += "run_0/"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    else:
        # if there are folders, find the last one and create a new one with incremented number
        existing_runs = [
            int(folder.name.split("_")[-1]) for folder in Path(checkpoint_path).iterdir() if folder.is_dir() and folder.name.startswith("run_")
        ]
        if existing_runs:
            last_run = max(existing_runs)
            checkpoint_path += f"run_{last_run + 1}/"

    num_of_evals = config["training"].get("num_of_evals", 10)

    training_args = SFTConfig(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        dataset_text_field="text",
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        eval_accumulation_steps=1,
        optim="adamw_8bit",
        max_steps=training_config["total_steps"],
        save_steps=training_config["total_steps"] // num_of_evals,
        logging_steps=training_config["total_steps"] // num_of_evals,
        eval_strategy="steps",
        eval_steps=training_config["total_steps"] // num_of_evals,
        save_total_limit=5,
        warmup_ratio=training_config["warmup_ratio"],
        learning_rate=training_config["learning_rate"],
        weight_decay=0.005,
        seed=training_config["seed_num"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=training_config["max_length"],
        bf16=is_bfloat16_supported(),
        dataset_num_proc=4, # for windows, best set to 1
        torch_compile=True, 
        torch_empty_cache_steps=training_config["total_steps"] // num_of_evals + 1,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
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
    eos_token = tokenizer.eos_token
    train_dataset, eval_dataset = prepare_datasets(config, eos_token)
    collator = get_collator(tokenizer)
    trainer, checkpoint_path = get_trainer(
        config, model, tokenizer, collator, train_dataset, eval_dataset
    )

    # write config to checkpoint directory
    config_path = checkpoint_path + "/config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

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
