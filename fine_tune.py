from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from config import HF_TOKEN

seed_num = 97

max_length = 12000


model = AutoModelForCausalLM.from_pretrained(
    "./llama-3.2-1B",  # token=HF_TOKEN
)  # temporary use 1B

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

tokenizer = AutoTokenizer.from_pretrained(
    "./llama-3.2-1B",
    max_length=max_length,
    # token=HF_TOKEN,
)  # temporary use 1B

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the dataset from huggingface dataset saved in hf_cognates_dataset
data = load_from_disk("hf_cognates_dataset").shuffle(seed=seed_num)
# test_data = load_from_disk("hf_cognates_test_dataset").shuffle(seed=seed_num)
val_inds = data.num_rows // 10
train_data = data.select(range(val_inds, data.num_rows))
val_data = data.select(range(0, val_inds))
val_inds = data.num_rows // 10

del data

# Need to double check the LoraConfig arguments
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


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
    output_dir="./sft_1b",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=500,
    learning_rate=5e-6,
    num_train_epochs=1,
    seed=seed_num,
    label_smoothing_factor=0.05,
    neftune_noise_alpha=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_seq_length=max_length,
    torch_compile=True,
    # bf16=True,
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
best_model_path = "./sft_1b_best"

# Save the model
trainer.model.save_pretrained(best_model_path)
