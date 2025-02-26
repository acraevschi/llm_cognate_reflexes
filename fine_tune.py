from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_from_disk

# from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import torch

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

seed_num = 97

max_length = 12288


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_length,
    dtype=None,
    load_in_4bit=True,
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the dataset from huggingface dataset saved in hf_cognates_dataset
data = load_from_disk("hf_cognates_dataset").shuffle(seed=seed_num)
# test_data = load_from_disk("hf_cognates_test_dataset").shuffle(seed=seed_num)
val_inds = data.num_rows // 10
train_data = data.select(range(val_inds, data.num_rows))
val_data = data.select(range(0, val_inds))

del data

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=32,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=seed_num,
    max_seq_length=max_length,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


def formatting_prompts_func(example):
    return f"{example['input']} {example['output']}"


train_data = train_data.map(lambda ex: {"text": formatting_prompts_func(ex)})
val_data = val_data.map(lambda ex: {"text": formatting_prompts_func(ex)})

instruction_template = "<Cognates>\n"
response_template = " <Prediction>\n"

collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    # padding_free=True,
    mlm=False,
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluations with no improvement
    early_stopping_threshold=0.0,  # Minimum change to qualify as an improvement
)

training_args = SFTConfig(
    output_dir="./sft_8b",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=32,
    eval_accumulation_steps=32,
    optim="adamw_8bit",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=500,
    warmup_ratio=0.1,
    learning_rate=5e-6,
    num_train_epochs=2,
    seed=seed_num,
    label_smoothing_factor=0.05,
    neftune_noise_alpha=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_seq_length=max_length,
    # torch_compile=True,
    bf16=is_bfloat16_supported(),
    dataloader_num_workers=8,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    formatting_func=None,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[early_stopping],
    # max_length=None,
)

trainer.train()

# Define the path to save the best model
best_model_path = "./sft_8b_best"

# Save the model
trainer.save_model(best_model_path)
