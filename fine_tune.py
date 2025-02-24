# TODO: read in dataset, prepare llama 3.1 8B and its tokenizer, tokenize data, randomly split 10% for validation, prepare lora, prepare SFTTrainer, train model (only on completions), evaluate model, save model

from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from config import HF_TOKEN

seed_num = 97

max_length = 32000


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-8B", token=HF_TOKEN
)  # temporary use 1B

# needed for gradient_checkpointing to work with LoRA
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

else:

    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-8B", max_length=max_length, token=HF_TOKEN
)  # temporary use 1B

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Load the dataset from huggingface dataset saved in hf_cognates_dataset
data = load_from_disk("hf_cognates_dataset").shuffle(seed=seed_num)
# test_data = load_from_disk("hf_cognates_test_dataset").shuffle(seed=seed_num)
val_inds = data.num_rows // 10
train_data = data.select(range(val_inds, data.num_rows))
val_data = data.select(range(val_inds))
del data

# Need to double check the LoraConfig arguments
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["input"])):
        text = f"{example['input'][i]} {example['output'][i]}"
        output_texts.append(text)
    return output_texts


instruction_template = "<NEWICK>"
response_template = " <Prediction>\n"

collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,
)

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluations with no improvement
    early_stopping_threshold=0.0,  # Minimum change to qualify as an improvement
)

training_args = SFTConfig(
    output_dir="./model_output",
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    # gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=400,
    save_steps=400,
    logging_steps=400,
    learning_rate=5e-6,
    num_train_epochs=2,
    seed=seed_num,
    label_smoothing_factor=0.05,
    neftune_noise_alpha=5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    max_seq_length=max_length,
    torch_compile=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    formatting_func=formatting_prompts_func,
    train_dataset=train_data,
    eval_dataset=val_data,
    callbacks=[early_stopping],
    # max_length=None,
)

trainer.train()
