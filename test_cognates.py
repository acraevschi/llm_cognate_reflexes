import argparse
import csv
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from hf_token import HF_TOKEN
from peft import PeftModel
from Levenshtein import distance as levenshtein_distance

# --- Configuration & Setup ---

PROMPT_BASE = """Comparative Linguistics Reconstruction Data

== Context ==
Evidence Data:
{evidence}

== Task Configuration ==
Target Language: {target_lang}

== Input Query ==
{query}

== Reconstructed Form ==
"""

def load_config(config_path):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        exit(1)

def normalized_edit_distance(prediction, target):
    """
    Calculates NED: Levenshtein(pred, target) / max(len(pred), len(target))
    """
    # Normalize strings: strip whitespace, lower case (optional, depending on strictness)
    pred_clean = prediction.strip()
    target_clean = target.strip()

    if len(pred_clean) == 0 and len(target_clean) == 0:
        return 0.0
    
    dist = levenshtein_distance(pred_clean, target_clean)
    max_len = max(len(pred_clean), len(target_clean))
    
    return dist / max_len if max_len > 0 else 1.0

# --- Data Preparation ---

def load_test_data(config):
    """Reconstructs the path to the test dataset based on config."""
    dataset_config = config["dataset"]
    dataset_name = f"{dataset_config['langs_per_entry']}langs_{dataset_config['num_evidence_sets']}evidence"
    test_data_path = os.path.join(dataset_config["output_test_path"], dataset_name)
    
    print(f"Loading test dataset from: {test_data_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Dataset not found at {test_data_path}")
        
    return load_from_disk(test_data_path)

# --- Metric 1: Loss Calculation ---

def calculate_test_loss(model, tokenizer, dataset, batch_size=1):
    """
    Calculates the Cross Entropy Loss on the test set.
    We manually batch this to avoid initializing a full Trainer instance just for eval.
    """
    print("Calculating Test Loss...")
    model.eval()
    
    total_loss = 0
    total_batches = 0
    
    # Pre-tokenize dataset for Loss calculation (Prompt + Completion)
    def tokenize_function(example):
        full_text = PROMPT_BASE.format(
            evidence=example['evidence'],
            query=example['query'],
            target_lang=example['target_lang']
        ) + f"{example['output']}" + tokenizer.eos_token
        
        return tokenizer(full_text, truncation=True, max_length=3584, padding="max_length")

    tokenized_ds = dataset.map(tokenize_function, batched=False, remove_columns=dataset.column_names)
    tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    dataloader = torch.utils.data.DataLoader(tokenized_ds, batch_size=batch_size)
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Loss Evaluation"):
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            
            # For CausalLM, labels are input_ids. The model handles shifting internally.
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item()
            total_batches += 1
            
    return total_loss / total_batches

# --- Metric 2: Generation & NED ---

def generate_and_calculate_ned(model, tokenizer, dataset, batch_size=8, show_samples=False):
    """
    Generates predictions and calculates Normalized Edit Distance.
    """
    print("Calculating Normalized Edit Distance (Generation)...")
    model.eval()
    
    ned_scores = []
    predictions = []
    references = []
    
    # Batch inputs
    batches = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
    
    for batch in tqdm(batches, desc="Generating & Scoring"):
        prompts = []
        ground_truths = batch['output']
        
        # Format prompts (without the answer)
        for i in range(len(ground_truths)):
            text = PROMPT_BASE.format(
                evidence=batch['evidence'][i],
                query=batch['query'][i],
                target_lang=batch['target_lang'][i]
            )
            prompts.append(text)
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            # Generate
            # max_new_tokens is low because cognates are usually single words
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=32, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False, # Greedy decoding for deterministic evaluation
                top_p=None,         # Disable nucleus sampling
                top_k=None,         # Disable top-k sampling
                temperature=None    # Disable temperature scaling
            )
        
        # Decode
        # We slice [input_len:] to get only the generated part
        input_lengths = inputs.input_ids.shape[1]
        generated_tokens = generated_ids[:, input_lengths:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate Metrics
        for i, (pred, ref) in enumerate(zip(decoded_preds, ground_truths)):
            # Clean up generation (sometimes models output newlines or extra spaces)
            clean_pred = pred.split('\n')[0].strip() 
            ned = normalized_edit_distance(clean_pred, ref)
            if show_samples and (i % 20 == 0):
                print(f"\nExample {i} | Pred: '{clean_pred}' | Ref: '{ref}' | NED: {ned:.4f}\n")
            ned_scores.append(ned)
            predictions.append(clean_pred)
            references.append(ref)

    return np.mean(ned_scores), predictions, references

# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-tuned Model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to original config file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--adapter_path", type=str, help="Path to the saved LoRA adapter (e.g., checkpoints/run_0/best_model)")
    group.add_argument("--model_path", type=str, help="Path to a full model checkpoint to load (e.g., checkpoints/run_0/full_model)")
    parser.add_argument("--show_samples", action="store_true", default=False, help="Whether to print sample reconstructions during evaluation")
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Determine tokenizer base (use full model checkpoint if provided, otherwise config model_name)
    tokenizer_source = args.model_path if args.model_path is not None else config["training"]["model_name"]

    # 1. Load Tokenizer
    print(f"Loading tokenizer from: {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        padding_side="left", # Left padding is usually better for generation
        token=HF_TOKEN
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare dtype & quant config
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",       # Standard for QLoRA
        bnb_4bit_compute_dtype=torch.bfloat16,  # match adapter
        bnb_4bit_use_double_quant=True,
    )

    # 3. Load model (either full model checkpoint or base model + adapter)
    if args.model_path is not None:
        print(f"Loading full model from: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=bnb_config,
            token=HF_TOKEN
        )
        print("Loaded full model checkpoint.")
    else:
        # adapter path provided
        model_name = config["training"]["model_name"]
        print(f"Loading base model: {model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            quantization_config=bnb_config,
            token=HF_TOKEN
        )
        print("Loaded base model; attaching adapter.")
        print(f"Loading adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)

    # 4. Load Data
    test_dataset = load_test_data(config)
    print(f"Test set size: {len(test_dataset)}")
    
    # 5. Calculate NED (Generation)
    tokenizer.padding_side = "left" # Switch back to left for generation
    avg_ned, preds, refs = generate_and_calculate_ned(
        model, 
        tokenizer, 
        test_dataset,
        batch_size=16, # hard-coded for now
        show_samples=args.show_samples
        )
    
    # 6. Report
    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Avg NED:          {avg_ned:.4f} (Lower is better)")
    print("="*30)
    
    # Optional: Inspect a few examples
    print("\nSample Reconstructions:")
    for i in range(min(5, len(preds))):
        print(f"Ref: {refs[i]:<20} | Pred: {preds[i]}")

    # Save metrics CSV next to the provided checkpoint's head directory
    save_source = args.adapter_path if args.adapter_path is not None else args.model_path
    head_dir = os.path.dirname(save_source)
    os.makedirs(head_dir, exist_ok=True)
    ckpt_name = os.path.basename(save_source)
    csv_path = os.path.join(head_dir, f"metrics_test_{ckpt_name}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["avg_ned", f"{avg_ned:.6f}"])
        writer.writerow(["num_examples", len(test_dataset)])
        writer.writerow(["checkpoint", ckpt_name])

    print(f"Saved metrics to: {csv_path}")

if __name__ == "__main__":
    main()