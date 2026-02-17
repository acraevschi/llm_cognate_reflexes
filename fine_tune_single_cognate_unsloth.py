import os
import json
import gc
import torch
from pathlib import Path
from datasets import load_from_disk

from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig

# --- Configuration & Setup ---


def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Config file {config_path} not found.")
        return None


def load_model_and_tokenizer(config):
    """
    Load Gemma 3 4B using Unsloth for efficient 4-bit training.
    """
    model_name = config["training"]["model_name"]
    max_seq_length = config["training"].get("max_length", 2048)

    print(f"Loading {model_name} via Unsloth...")

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        dtype=None,
    )

    # Add LoRA adapters
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,  # Turn off for just text!
        finetune_language_layers=True,  # Should leave on!
        finetune_attention_modules=True,  # Attention good for GRPO
        finetune_mlp_modules=True,  # Should leave on always!
        r=32,  # Larger = higher accuracy, but might overfit
        lora_alpha=64,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=1997,
    )

    # Setup Gemma 3 specific chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-3",
    )

    print("Model and Tokenizer loaded successfully!\n")
    return model, tokenizer


# --- Data Preparation ---


def format_cognate_prompt(evidence, query):
    """Constructs the user instruction."""
    return f"""Reconstruct cognates based on the provided evidence.
    
### Evidence:
{evidence}

### Target Context:
{query}
"""


def formatting_prompts_func(examples, tokenizer):
    """
    Formats the dataset into Gemma-3 chat format.
    Combines 'evidence' + 'query' into User message.
    Uses 'output' (target_form) as Model message.
    """
    convos = []

    # Iterate through the batch
    for evidence, query, output in zip(
        examples["evidence"], examples["query"], examples["output"]
    ):
        # Create the conversation structure
        conversation = [
            {"role": "user", "content": format_cognate_prompt(evidence, query)},
            {"role": "assistant", "content": output},
        ]
        convos.append(conversation)

    # Apply the chat template
    # IMPORTANT: We remove <bos> because the SFTTrainer/Collator adds it automatically.
    # If we don't remove it, we get double BOS tokens.
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        ).removeprefix("<bos>")
        for convo in convos
    ]

    return {"text": texts}


def prepare_datasets(config, tokenizer):
    """Load and process the dataset for SFT"""
    dataset_config = config["dataset"]
    # Adjust path logic based on your specific directory structure
    dataset_string = f"{dataset_config['langs_per_entry']}langs_{dataset_config['num_evidence_sets']}evidence"
    train_data_path = f"{dataset_config['output_train_path']}/{dataset_string}"

    print(f"Loading dataset from: {train_data_path}")
    data = load_from_disk(train_data_path)

    # Shuffle
    data = data.shuffle(seed=1997)

    # Split (Standard SFT usually requires a split for eval loss)
    eval_percent = config["training"].get("eval_percent", 10)
    eval_inds = data.num_rows // eval_percent
    train_data = data.select(range(eval_inds, data.num_rows))
    eval_data = data.select(range(0, eval_inds))

    print("Formatting datasets for Chat (SFT)...")

    # Apply formatting
    fn_kwargs = {"tokenizer": tokenizer}

    train_data = train_data.map(
        formatting_prompts_func,
        batched=True,
        fn_kwargs=fn_kwargs,
        num_proc=4,
    )
    eval_data = eval_data.map(
        formatting_prompts_func,
        batched=True,
        fn_kwargs=fn_kwargs,
        num_proc=4,
    )

    return train_data, eval_data


# --- Training ---


def train_model(config_path="config.json"):
    config = load_config(config_path)
    if not config:
        return

    # 1. Load Model
    model, tokenizer = load_model_and_tokenizer(config)

    # 2. Prepare Data
    train_dataset, eval_dataset = prepare_datasets(config, tokenizer)

    # 3. Setup Output Directory
    training_config = config["training"]
    # Shorten model name for folder structure
    output_dir = f"{training_config['checkpoint_dir']}/gemma3-sft"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 4. Define Training Arguments (SFTConfig)
    # Mapping your original config values to SFTConfig
    args = SFTConfig(
        output_dir=output_dir,
        eval_on_start=True,
        per_device_train_batch_size=training_config.get("batch_size", 2),
        per_device_eval_batch_size=training_config.get("batch_size", 2),
        gradient_accumulation_steps=training_config.get(
            "gradient_accumulation_steps", 4
        ),
        eval_accumulation_steps=1,
        learning_rate=training_config.get("learning_rate", 2e-4),
        max_steps=training_config.get(
            "total_steps", 1000
        ),  # Unsloth typically uses max_steps or num_epochs
        logging_steps=training_config["total_steps"] // training_config["num_of_evals"],
        save_steps=training_config["total_steps"] // training_config["num_of_evals"],
        eval_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eval_steps=training_config["total_steps"] // training_config["num_of_evals"],
        warmup_ratio=training_config["warmup_ratio"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        optim="adamw_8bit",
        weight_decay=0.1,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=1997,
        dataset_text_field="text",  # The field we created in formatting_prompts_func
        max_seq_length=config["training"].get("max_length", 2048),
        report_to="none",  # Change to "wandb" if needed
        packing=False,  # Can be set to True for speedup if sequences are short
        torch_empty_cache_steps=training_config["total_steps"]
        // (training_config["num_of_evals"] * 2),
    )

    # 5. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=training_config.get(
                    "early_stopping_patience", 3
                )
            )
        ],
    )

    # 6. Apply Unsloth's "Train on Responses Only"
    # This masks the user instruction so the model doesn't learn to generate the prompt
    trainer = train_on_responses_only(
        trainer=trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("\n\n---Starting SFT training---\n\n")
    trainer_stats = trainer.train()
    print("\n\n---Training complete!---\n\n")

    # 7. Save Model
    # Save the LoRA adapters
    final_path = os.path.join(output_dir, "final_adapter")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save config copy
    with open(os.path.join(final_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)

    print(f"Adapters saved to: {final_path}")

    # Clean up
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    train_model()
