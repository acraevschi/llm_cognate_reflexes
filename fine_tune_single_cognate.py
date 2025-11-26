from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from torch.cuda import empty_cache
import os
import torch
import gc
import json
from pathlib import Path
from metrics import calculate_ned
import numpy as np

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

instruction_template = "### Instruction:"
input_template = "### Evidence Sets (Reference Data):"
response_template = "### Output:"

prompt_base = """### Instruction:
You are a historical linguist. Your task is to reconstruct the missing word(s) form for the **Target Language** in the **Query Set**. The target word(s) will be marked as ???\n
Use the **Evidence Sets** provided below to identify regular sound correspondences and phonological patterns between the languages.

### Evidence Sets (Reference Data):
{evidence}
### Query Set (Task):
{query}
### Target Language:
{target_lang}

### Output:
"""

class GenerativeEvalCallback(TrainerCallback):
    """
    Custom callback to generate text and calculate NED on a subset of validation data
    at the end of every epoch (or specified steps).
    """
    def __init__(self, model, tokenizer, eval_dataset, num_samples=None, response_template=response_template):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = len(eval_dataset) if not num_samples else num_samples
        self.response_template = response_template

    def on_evaluate(self, args, state, control, **kwargs):
        print("\nRunning Generative NED Evaluation...")
        
        # Enable inference mode
        FastLanguageModel.for_inference(self.model)
        
        # Select random samples
        if self.num_samples and self.num_samples < len(self.eval_dataset):
            indices = np.random.choice(len(self.eval_dataset), self.num_samples, replace=False)
            samples = self.eval_dataset.select(indices)
        else: 
            samples = self.eval_dataset
        
        ned_scores = []
        
        for sample in samples:
            # The sample['text'] contains: Prompt + Response Template + Target + EOS
            full_text = sample['text']
            
            # Split into Prompt (Input) and Target (Ground Truth)
            split_text = full_text.split(self.response_template)
            
            if len(split_text) < 2: 
                continue # Skip malformed examples
            
            # Reconstruct the prompt part (everything before the output)
            input_prompt = split_text[0] + self.response_template
            
            # The Ground Truth is everything after the template, minus the EOS token
            ground_truth = split_text[1].replace(self.tokenizer.eos_token, "").strip()
            
            # Tokenize and Generate
            inputs = self.tokenizer([input_prompt], return_tensors="pt").to("cuda")
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=64, # Cognates are usually short, 64 is plenty
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Extract only the generated part
            # The model output includes the input_prompt. We split to get the new text.
            generated_response = decoded_output.split(self.response_template)[-1].strip()
            
            # Calculate Metric (Direct string comparison now)
            ned = calculate_ned(generated_response, ground_truth)
            ned_scores.append(ned)
            
            ### Optional: Print first few examples to debug
            # if len(ned_scores) <= 3:
            #     print(f"\n[Ex {len(ned_scores)}]")
            #     print(f"Target Lang: {sample.get('target_lang', 'Unknown')}")
            #     print(f"Input Context (snippet): ...{input_prompt[-100:].replace(chr(10), ' ')}")
            #     print(f"Pred: '{generated_response}' | True: '{ground_truth}' | NED: {ned:.4f}")
        
        avg_ned = sum(ned_scores) / len(ned_scores) if ned_scores else 0.0
        
        print(f"\n==========================================")
        print(f"Validation NED (sample size {len(ned_scores)}): {avg_ned:.4f}")
        print(f"==========================================\n")
        
        if self.trainer:
            self.trainer.log({"eval_ned": avg_ned})

        # Reset model to training mode
        FastLanguageModel.for_training(self.model)

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
            load_in_4bit=False,
            attn_implementation="flash_attention_2",
        )
    except:
        print("Flash attention not working on this machine, trying without it...\n")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_length,
            dtype=torch.bfloat16,
            load_in_4bit=False,
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
    # tokenizer.padding_side = "right"

    return model, tokenizer


def formatting_prompts_func(example, eos_token):
    """
    Constructs the full prompt using the fields created in compile_dataset.py
    """
    # Fill the template
    input_text = prompt_base.format(
        evidence=example['evidence'],
        query=example['query'],
        target_lang=example['target_lang']
    )
    
    # Combine Input + Target Output
    # We strip whitespace to ensure clean concatenation
    full_text = f"{input_text}{example['output']}{eos_token}"
    
    return full_text


def get_collator(tokenizer):
    return DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )


def prepare_datasets(config, eos_token, eval_percent=10):
    """Load and prepare the dataset"""
    dataset_config = config["dataset"]
    # Ensure name matches what was saved in compile_dataset
    dataset_string = f"{dataset_config['langs_per_entry']}langs_{dataset_config['num_evidence_sets']}evidence"
    train_data_path = f"{dataset_config['output_train_path']}/{dataset_string}"

    print(f"Loading dataset from: {train_data_path}")
    data = load_from_disk(train_data_path)
    
    # Simple split if not pre-split, or just shuffle
    data = data.shuffle(seed=42)
    
    # Map the formatting function
    # Note: remove_columns is important here to free up memory and ensure the trainer 
    # only sees the 'text' column
    data = data.map(
        lambda ex: {"text": formatting_prompts_func(ex, eos_token)},
    )
    
    # OR just split the train data as before:
    eval_inds = data.num_rows // eval_percent
    train_data = data.select(range(eval_inds, data.num_rows))
    eval_data = data.select(range(0, eval_inds))
    
    return train_data, eval_data


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
        lr_scheduler_type=training_config["lr_scheduler_type"],
        weight_decay=0.01,
        seed=training_config["seed_num"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=training_config["max_length"],
        bf16=is_bfloat16_supported(),
        dataset_num_proc=1, # for windows, best set to 1
        torch_compile=True, # for windows, best set to False
        torch_empty_cache_steps=training_config["total_steps"] // num_of_evals + 1,
    )

    ned_callback = GenerativeEvalCallback(model, tokenizer, eval_dataset, num_samples=training_config.get("num_samples"))

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
            ned_callback,
            EarlyStoppingCallback(
                early_stopping_patience=training_config["early_stopping_patience"],
                early_stopping_threshold=training_config["early_stopping_threshold"],
            ),
        ],
    )

    ned_callback.trainer = trainer

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
    train_dataset, eval_dataset = prepare_datasets(config, eos_token, eval_percent=config["training"]["eval_percent"])
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
