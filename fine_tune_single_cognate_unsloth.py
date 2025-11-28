from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from transformers import EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
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
    at the end of every epoch. 
    Refactored to work with 'prompt' and 'completion' dictionary format.
    """
    def __init__(self, model, tokenizer, eval_dataset, num_samples=None):
        self.model = model
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.num_samples = len(eval_dataset) if not num_samples else num_samples

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
            # DIRECT ACCESS: No need to split strings anymore.
            # The dataset now contains specific 'prompt' and 'completion' keys.
            input_prompt = sample['prompt']
            
            # Ground truth is the completion (remove EOS for strict string comparison)
            ground_truth = sample['completion'].replace(self.tokenizer.eos_token, "").strip()
            
            # Tokenize and Generate
            inputs = self.tokenizer([input_prompt], return_tensors="pt").to("cuda")
            
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # The model outputs [Prompt + Generated]. We need to strip the prompt.
            # Since decoded_output includes the input_prompt text, we can slice or split.
            # A robust way is to split by the prompt end, or simply use the known length.
            # Here we split by response_template since it's the anchor at the end of the prompt.
            if response_template in decoded_output:
                generated_response = decoded_output.split(response_template)[-1].strip()
            else:
                # Fallback if template generation is malformed, just take the raw diff
                generated_response = decoded_output[len(input_prompt):].strip()
            
            # Calculate Metric
            ned = calculate_ned(generated_response, ground_truth)
            ned_scores.append(ned)
        
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
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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

    return model, tokenizer


def formatting_prompts_func(example, eos_token):
    """
    Constructs the dictionary with 'prompt' and 'completion' keys.
    """
    # 1. Build the Prompt (Instruction + Evidence + Query + Target Lang + Output Header)
    input_text = prompt_base.format(
        evidence=example['evidence'],
        query=example['query'],
        target_lang=example['target_lang']
    )
    
    # 2. Get the Completion (Output + EOS)
    completion_text = f"{example['output']}{eos_token}"
    
    # Return dictionary required for standard prompt-completion training
    return {
        "prompt": input_text,
        "completion": completion_text
    }


def prepare_datasets(config, eos_token, eval_percent=10):
    """Load and prepare the dataset into prompt/completion columns"""
    dataset_config = config["dataset"]
    dataset_string = f"{dataset_config['langs_per_entry']}langs_{dataset_config['num_evidence_sets']}evidence"
    train_data_path = f"{dataset_config['output_train_path']}/{dataset_string}"

    print(f"Loading dataset from: {train_data_path}")
    data = load_from_disk(train_data_path)
    
    data = data.shuffle(seed=42)
    
    # Map to new structure and remove old columns to ensure SFTTrainer picks up the right ones
    original_columns = data.column_names
    data = data.map(
        lambda ex: formatting_prompts_func(ex, eos_token),
        remove_columns=original_columns 
    )
    
    # Split train/eval
    eval_inds = data.num_rows // eval_percent
    train_data = data.select(range(eval_inds, data.num_rows))
    eval_data = data.select(range(0, eval_inds))
    
    return train_data, eval_data


def get_trainer(config, model, tokenizer, train_dataset, eval_dataset):
    """Create the trainer with config parameters"""
    training_config = config["training"]
    model_name = training_config["model_name"].split("/")[1]
    
    checkpoint_path = (
        f"{training_config['checkpoint_dir']}/{model_name}/"
    )

    if not os.path.exists(checkpoint_path):
        checkpoint_path += "run_0/"
        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    else:
        existing_runs = [
            int(folder.name.split("_")[-1]) for folder in Path(checkpoint_path).iterdir() if folder.is_dir() and folder.name.startswith("run_")
        ]
        if existing_runs:
            last_run = max(existing_runs)
            checkpoint_path += f"run_{last_run + 1}/"

    num_of_evals = config["training"].get("num_of_evals", 10)

    # Use SFTConfig with completion_only_loss kwargs
    training_args = SFTConfig(
        output_dir=checkpoint_path,
        overwrite_output_dir=True,
        completion_only_loss=True,
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
        dataset_num_proc=4,
        torch_compile=True,
        torch_empty_cache_steps=training_config["total_steps"] // num_of_evals + 1
    )

    ned_callback = GenerativeEvalCallback(model, tokenizer, eval_dataset, num_samples=training_config.get("num_samples"))

    # Create trainer without explicit DataCollator
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
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

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    empty_cache()
    gc.collect()

    model, tokenizer = load_model_and_tokenizer(config)
    eos_token = tokenizer.eos_token
    
    # Prepare data with prompt/completion columns
    train_dataset, eval_dataset = prepare_datasets(config, eos_token, eval_percent=config["training"]["eval_percent"])
    
    # Get trainer (no collator passed)
    trainer, checkpoint_path = get_trainer(
        config, model, tokenizer, train_dataset, eval_dataset
    )

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