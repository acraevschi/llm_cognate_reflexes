import os
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,  # prepare_model_for_kbit_training
    PeftModel,
)
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence

BASE_MODEL = r"C:\Users\luca\Desktop\Deepseek\FineTune\DeepSeek-R1-Distill-Llama-8B"
DATASET_PATH = r"C:\Users\luca\Desktop\Deepseek\FineTune\dataset.jsonl"
OUTPUT_DIR = r"C:\Users\luca\Desktop\Deepseek\FineTune\Deepseek-R1-FineTuned"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

try:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
except Exception as e:
    print(f"Error loading model: {e}")
    from transformers import clear_pretrained_cache

    clear_pretrained_cache()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

data = load_dataset("json", data_files=DATASET_PATH)
train_val = data["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
train_data = train_val["train"]
val_data = train_val["test"]

NUM_TRAIN_SAMPLES = 100
NUM_VAL_SAMPLES = 20

train_data = train_data.select(range(min(NUM_TRAIN_SAMPLES, len(train_data))))
val_data = val_data.select(range(min(NUM_VAL_SAMPLES, len(val_data))))

prompt_without_output = (
    "Below is an instruction that describes a task, paired with a question that provides further context.\n"
    "Write a response that appropriately answers the question.\n"
    "REMEMBER: ALWAYS start your output with `<think>` and reason step by step. Break down the problem, consider alternatives, and validate your assumptions. Provide brief examples to illustrate your reasoning. Once you finish your internal reasoning, output `</think>` and then provide a detailed and accurate final answer that directly addresses the user's query.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Question:\n{input}\n\n"
    "### Response:\n"
)


def formatting_prompts_func(examples):
    prompts = []
    full_texts = []
    for instr, inp, out in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        prompt = prompt_without_output.format(instruction=instr, input=inp)
        full_text = prompt + out + tokenizer.eos_token
        prompts.append(prompt)
        full_texts.append(full_text)
    return {"prompt": prompts, "text": full_texts}


train_data = train_data.map(formatting_prompts_func, batched=True)
val_data = val_data.map(formatting_prompts_func, batched=True)


def tokenize_function(examples):
    tokenized_full = tokenizer(examples["text"], truncation=True, max_length=512)
    tokenized_prompt = tokenizer(examples["prompt"], truncation=True, max_length=512)

    labels = []
    for full_ids, prompt_ids in zip(
        tokenized_full["input_ids"], tokenized_prompt["input_ids"]
    ):
        prompt_len = len(prompt_ids)
        label = full_ids.copy()
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)
    tokenized_full["labels"] = labels
    return tokenized_full


train_data = train_data.map(
    tokenize_function, batched=True, remove_columns=train_data.column_names
)
val_data = val_data.map(
    tokenize_function, batched=True, remove_columns=val_data.column_names
)


def custom_data_collator(features):
    input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
    attention_masks = [
        torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
    ]
    labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
    batch_input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    batch_attention_masks = pad_sequence(
        attention_masks, batch_first=True, padding_value=0
    )
    batch_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_masks,
        "labels": batch_labels,
    }


model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=100,
    max_steps=500,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    optim="adamw_torch",
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=False,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=custom_data_collator,
)

model.config.use_cache = False
model = torch.compile(model)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="cpu",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
model = model.to("cpu")

merged_model = model.merge_and_unload()
merged_model = merged_model.to("cpu")

merged_output_dir = os.path.join(OUTPUT_DIR, "merged_model")
merged_model.save_pretrained(merged_output_dir, safe_serialization=True)
tokenizer.save_pretrained(merged_output_dir)
print(f"Merged and full model saved to: {merged_output_dir}")
