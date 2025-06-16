from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from torch.cuda import empty_cache
import os
import torch
import gc
import numpy as np
from pathlib import Path

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

empty_cache()
gc.collect()

seed_num = 97
max_length = 4096
total_steps = 240
early_stopping_patience = 4
early_stopping_threshold = 0.05
checkpoint_dir = "./checkpoints/sft_qwen3_1_7b_run2"

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

# Make sure the checkpoint directory exists
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

def load_model_and_tokenizer(model_path="unsloth/Qwen3-1.7B-unsloth-bnb-4bit"):
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
            attn_implementation="flash_attention_2",
        )
    except:
        print("Flash attention not working on this machine, trying without it...\n")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
    print("The model was successfully loaded!\n")

    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj",
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

def prepare_datasets(train_data_path="datasets/hf_cognates_dataset_25concepts_min8_50combs"):
    # Load and prepare the dataset
    data = load_from_disk(train_data_path)

    val_inds = data.num_rows // 10
    train_data = data.select(range(val_inds, data.num_rows))
    val_data = data.select(range(0, val_inds))
    
    # Format the datasets
    train_data = train_data.map(lambda ex: {"text": formatting_prompts_func(ex)})
    val_data = val_data.map(lambda ex: {"text": formatting_prompts_func(ex)})
    
    return train_data, val_data

def get_trainer(model, collator, train_dataset, eval_dataset):
    training_args = SFTConfig(
        output_dir=checkpoint_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=16,
        eval_accumulation_steps=2,
        optim="adamw_8bit",
        max_steps=total_steps,  # Use max_total_steps
        save_steps=total_steps//8,
        logging_steps=total_steps//8,
        eval_strategy="steps", # Evaluate at each save_steps
        eval_steps=total_steps//8, # Evaluate at each save_steps
        save_total_limit=4,
        warmup_ratio=0.1,
        learning_rate=1e-4,
        weight_decay=0.01,
        seed=seed_num,
        load_best_model_at_end=True,  # Load best model at the end
        metric_for_best_model="loss", # Can be loss or eval_loss depending on trainer
        greater_is_better=False,
        max_seq_length=max_length,
        bf16=is_bfloat16_supported(),
        dataset_num_proc=1,
        torch_compile=True,
        torch_empty_cache_steps=total_steps//8 + 1, # empty cache after evaluation
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        formatting_func=None,  # Already handled by dataset mapping
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[
            EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience, 
                    early_stopping_threshold=early_stopping_threshold
                )
        ],
    )
    return trainer

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    train_dataset, eval_dataset = prepare_datasets()
    collator = get_collator(tokenizer)
    trainer = get_trainer(model, collator, train_dataset, eval_dataset)
    print("\n\n---Starting training---\n\n")
    trainer.train()
    print("\n\n---Training complete!---\n\n")
    path_best_model = checkpoint_dir + "/best_model"
    best_model_path = trainer.save_model(path_best_model)
    tokenizer.save_pretrained(path_best_model)
    empty_cache()
    gc.collect()

    print(f"Best model saved at: {path_best_model}")
