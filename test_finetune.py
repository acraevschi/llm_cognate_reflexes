import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

with open("prompt.txt", "r") as file:
    reconstruction_prompt = file.read()


model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

# 1. Prepare Dataset
data_prompts = [
    """A: [dalaa, bigu, maroa, zanii, gobaa, zivaa]
    B: [tal, fex, mar, san, kop, -]
    C: [sal, fiʃ, mar, sen, ʃop, -]"""
]  # List of your data prompts

correct_outputs = [
    """Proto-forms: [*tala, *piku, *maro, *sani, *kopa, *sini]
    A: [dalaa, bigu, maroa, zanii, gobaa, zivaa]
    B: [tal, fex, mar, san, kop, +sen]
    C: [sal, fiʃ, mar, sen, ʃop, +sin]"""
]  # List of your correct reconstructions

# 2. Tokenization and Processing
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dict_dataset = Dataset.from_dict(
    {
        "instruction": [reconstruction_prompt + "\n" + data for data in data_prompts],
        "output": correct_outputs,
    }
)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        text = f"### Instruction: {example['instruction'][i]}\n  ###Reconstructed forms: {example['output'][i]}"
        output_texts.append(text)
    return output_texts


response_template = "  ###Reconstructed forms:"

collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# Prepare the base model with LoRA config
lora_config = LoraConfig(
    r=8,  # Rank - smaller rank for smaller dataset
    lora_alpha=32,  # Alpha parameter for LoRA scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target attention modules
    lora_dropout=0.1,  # Dropout probability for LoRA layers
    bias="none",  # We don't train bias terms
    task_type="CAUSAL_LM",  # Task type for causal language modeling
)

# Prepare model for training
model.gradient_checkpointing_enable()  # Enable gradient checkpointing
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)


# Training arguments optimized for small dataset
training_arguments = SFTConfig(
    output_dir="./results",
    run_name="test_run",
    max_seq_length=4096,  # Adjust based on your sequence length
    num_train_epochs=3,  # Start with 3 epochs for small dataset
    per_device_train_batch_size=2,  # Small batch size to save memory
    gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch
    learning_rate=2e-4,  # Slightly higher learning rate for LoRA
    logging_steps=10,  # Log more frequently for small dataset
    save_steps=50,  # Save checkpoints more frequently
    optim="paged_adamw_32bit",  # Memory-efficient optimizer
    warmup_ratio=0.03,  # Short warmup for small dataset
    lr_scheduler_type="cosine",  # Cosine schedule works well with LoRA
    # eval_strategy="no",    # Evaluate periodically
    # eval_steps=50,                # Evaluate every 50 steps
    # save_total_limit=3,           # Keep only last 3 checkpoints
    # load_best_model_at_end=True,  # Load the best model after training
    # Memory optimizations
    gradient_checkpointing=True,
    torch_compile=True,  # Use torch.compile for speedup
    bf16=True,  # Use bfloat16 for training
    report_to="none",
)

tokenizer.padding_side = "right"
# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dict_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
    args=training_arguments,
)

# Train the model
trainer.train(use_cache=True)

# Save the trained model
trainer.save_model("final_checkpoint")
