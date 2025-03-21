# Imports
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoConfig
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset
from pathlib import Path
import pickle
import os
from Bio import SeqIO
import time
import argparse
import gc

# Configure CUDA settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from transformers import TrainerCallback

class DetailedCallback(TrainerCallback):
    def __init__(self):
        self.step_start_time = time.time()
        
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\nTraining started at {time.strftime('%H:%M:%S')}")
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        print(f"\nStep {state.global_step} beginning")
        print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
    def on_step_end(self, args, state, control, **kwargs):
        step_time = time.time() - self.step_start_time
        print(f"Step {state.global_step} completed in {step_time:.2f}s")
        print(f"Learning rate: {self.trainer.optimizer.param_groups[0]['lr']}")

def compute_metrics_f1_score(eval_pred):
    """Computes F1 score for binary classification"""
    print("\nInside compute_metrics:")
    print(f"Predictions shape: {eval_pred.predictions.shape}")
    print(f"Labels shape: {eval_pred.label_ids.shape}")
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r={'f1_score': f1_score(references, predictions)}
    print(f"F1 score: {r['f1_score']}")
    return r

def collect_attention_scores(model, tokenizer, test_dataloader, device):
    """Collect attention scores from model outputs"""
    # Get model configuration
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # Initialize dictionary to store attention scores
    examples_scores_attention = {i: {j: [] for j in range(num_heads)} for i in range(num_layers)}
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get model outputs with attention
            outputs = model(
                input_ids, 
                attention_mask=attention_mask,
                output_attentions=True
            )
            all_attentions = outputs.attentions  # This contains attention for all layers
            
            # Convert input_ids to tokens
            input_tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids.tolist()]
            
            # Process each layer and head
            for layer in range(num_layers):
                for head in range(num_heads):
                    # Get attention scores for specific head
                    attention_scores_per_head = all_attentions[layer][:, head, :, :].detach()
                    
                    # Process each example in batch
                    for tokens, att_matrix in zip(input_tokens, attention_scores_per_head):
                        max_att_scores = att_matrix.max(dim=0)[0].detach().cpu().numpy()
                        examples_scores_attention[layer][head].append((max_att_scores, tokens))
    
    return examples_scores_attention

def save_attention_scores(examples_scores_attention, save_dir):
    """Save attention scores to specified directory"""
    os.makedirs(save_dir, exist_ok=True)
    
    for layer in examples_scores_attention:
        attention_filename = os.path.join(save_dir, f'examples_scores_attention_layer{layer}.p')
        with open(attention_filename, 'wb') as f:
            pickle.dump(examples_scores_attention[layer], f)

def parse_fasta(fasta_file):
    """Parse a FASTA file to extract sequences and labels based on format '|label' in description"""
    sequences = []
    labels = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        # Extract label from the record description using split('|')[-1]
        label = int(record.description.split('|')[-1])
        sequences.append(sequence)
        labels.append(label)
            
    return sequences, labels


def main():
    parser = argparse.ArgumentParser(description="Get attention scores for models")
    
    # Model selection flags
    parser.add_argument("--random", action="store_true", help="Run with randomly initialized model")
    parser.add_argument("--pretrained", action="store_true", help="Run with pretrained model")
    parser.add_argument("--finetuned", action="store_true", help="Run with finetuned model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint to use (e.g., 'checkpoint-1000')")
    
    # Dataset selection
    parser.add_argument("--dataset", type=str, choices=["CUSTOM", "FAKE", "ENHANCER"], 
                        required=True, help="Dataset type to use")
    
    # Other parameters
    parser.add_argument("--gpu", type=str, default="0", help="GPU device to use")
    parser.add_argument("--max_samples", type=int, default=1000, 
                        help="Maximum samples per class to use for attention analysis")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for attention collection")
    
    args = parser.parse_args()
    
    # If no model flags are set, run all models
    if not (args.random or args.pretrained or args.finetuned):
        args.random = True
        args.pretrained = True
        args.finetuned = True
        print("No model flags specified. Running all models.")
    
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set data paths and output directories based on dataset choice
    base_path = "/home/mica/nucleotide-transformer/data"
    
    if args.dataset == "CUSTOM":
        train_file = f"{base_path}/custom_TATA_train.fa"
        test_file = f"{base_path}/custom_TATA_test.fa"
        output_base = "/home/mica/nucleotide-transformer/custom_TATA/"
        checkpoint_dir = f"/home/mica/nucleotide-transformer/CUSTOM-nucleotide-transformer-finetuned-NucleotideTransformer"
    elif args.dataset == "FAKE":
        train_file = f"{base_path}/fake_TATA_train.fa"
        test_file = f"{base_path}/fake_TATA_test.fa"
        output_base = "/home/mica/nucleotide-transformer/fake_TATA/"
        checkpoint_dir = f"/home/mica/nucleotide-transformer/FAKE-nucleotide-transformer-finetuned-NucleotideTransformer"
    elif args.dataset == "ENHANCER":
        train_file = f"{base_path}/enhancer_train.fa"
        test_file = f"{base_path}/enhancer_test.fa"
        output_base = "/home/mica/nucleotide-transformer/enhancer/"
        checkpoint_dir = f"/home/mica/nucleotide-transformer/ENHANCER-nucleotide-transformer-finetuned-NucleotideTransformer"
    
    # Set checkpoint path if specified
    if args.checkpoint:
        full_checkpoint_path = f"{checkpoint_dir}/{args.checkpoint}"
    else:
        # Default checkpoint paths
        if args.dataset == "CUSTOM":
            full_checkpoint_path = f"{checkpoint_dir}/checkpoint-1000"
        elif args.dataset == "FAKE":
            full_checkpoint_path = f"{checkpoint_dir}/checkpoint-1000"
        elif args.dataset == "ENHANCER":
            full_checkpoint_path = f"{checkpoint_dir}/checkpoint-4000"
    
    # Create output directories
    output_dirs = {
        "random_init": os.path.join(output_base, "random_init/"),
        "pretrained": os.path.join(output_base, "pretrained/"),
        "finetuned": os.path.join(output_base, "finetuned/")
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created output directory: {dir_path}")

    # Load datasets
    print(f"Loading dataset: {args.dataset}")
    train_sequences, train_labels = parse_fasta(train_file)
    test_sequences, test_labels = parse_fasta(test_file)

    # Balance test dataset if needed
    if args.max_samples > 0:
        class_0_indices = [i for i, label in enumerate(test_labels) if label == 0][:args.max_samples]
        class_1_indices = [i for i, label in enumerate(test_labels) if label == 1][:args.max_samples]
        selected_indices = sorted(class_0_indices + class_1_indices)
        
        test_sequences = [test_sequences[i] for i in selected_indices]
        test_labels = [test_labels[i] for i in selected_indices]

    # Split training data to get validation set
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=0.1, random_state=42
    )

    print(f"Train samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    print(f"Test samples: {len(test_sequences)}")
    print(f"Class distribution in test: {test_labels.count(1)} class 1, {test_labels.count(0)} class 0")

    # Load model components
    num_labels = 2
    model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Create datasets
    ds_train = Dataset.from_dict({"data": train_sequences, "labels": train_labels})
    ds_val = Dataset.from_dict({"data": val_sequences, "labels": val_labels})
    ds_test = Dataset.from_dict({"data": test_sequences, "labels": test_labels})
    
    def tokenize_function(examples):
        # Use padding and truncation to handle variable sequence lengths
        outputs = tokenizer(
            examples["data"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return outputs

    # Tokenize datasets
    tokenized_train = ds_train.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_val = ds_val.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_test = ds_test.map(tokenize_function, batched=True, remove_columns=["data"])
    
    # ======== PART 1: RANDOM INITIALIZED MODEL ========
    if args.random:
        print("Processing randomly initialized model...")
        
        # Get config from pretrained model but initialize randomly
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels,
            output_attentions=False,
            trust_remote_code=True
        )

        print("Got config. Getting random model...")
        
        # Initialize random model
        load_time = time.time()
        random_model = AutoModelForSequenceClassification.from_config(
            config,
            trust_remote_code=True
        )
        load_time = time.time() - load_time

        print(f"Model loaded in {load_time:.2f} seconds")
        print("Reinitializing weights...")
        
        # Explicitly reinitialize weights
        def reinit_weights(model):
            for module in model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                    
        initialize_time = time.time()
        reinit_weights(random_model)
        random_model.to(device)
        initialize_time = time.time() - initialize_time

        print(f"Model ready to go in {initialize_time:.2f} seconds")
        
        # Prepare test dataloader with small batch size
        args_random = TrainingArguments(
            output_dir="./tmp_random",
            per_device_eval_batch_size=args.batch_size
        )
        
        trainer_random = Trainer(
            model=random_model,
            args=args_random
        )
        
        # Enable attentions for inference
        random_model.config.output_attentions = True
        random_model.config.return_dict = True
        
        # Get test dataloader
        test_dataloader = trainer_random.get_test_dataloader(tokenized_test)
        
        print("Starting attention score collection...")
        collection_start = time.time()
        # Get attention scores
        random_attention_scores = collect_attention_scores(
            random_model, tokenizer, test_dataloader, device
        )
        collection_time = time.time() - collection_start
        print(f"Attention scores collected in {collection_time:.2f} seconds")
        
        # Save attention scores
        save_attention_scores(random_attention_scores, output_dirs["random_init"])
        print(f"Saved random model attention scores to {output_dirs['random_init']}")
        
        # Clear model from GPU
        del random_model
        del trainer_random
        torch.cuda.empty_cache()
    
    # ======== PART 2: PRETRAINED MODEL ========
    if args.pretrained:
        print("Processing pretrained model...")
        
        # Load pretrained model
        print(f"Loading pretrained model from {model_name}")
        load_start = time.time()
        pretrained_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, output_attentions=False, trust_remote_code=True
        )
        load_time = time.time() - load_start
        print(f"Pretrained model loaded in {load_time:.2f} seconds")
        print(f"Moving model to {device}")
        pretrained_model.to(device)
        print("Model successfully moved to device")
        
        args_pretrained = TrainingArguments(
            output_dir="./tmp_pretrained",
            per_device_eval_batch_size=args.batch_size
        )
        
        trainer_pretrained = Trainer(
            model=pretrained_model,
            args=args_pretrained
        )
        
        # Enable attentions for inference
        pretrained_model.config.output_attentions = True
        pretrained_model.config.return_dict = True
        
        # Get test dataloader
        test_dataloader = trainer_pretrained.get_test_dataloader(tokenized_test)
        
        # Get attention scores
        print("Starting attention score collection...")
        collection_start = time.time()
        pretrained_attention_scores = collect_attention_scores(
            pretrained_model, tokenizer, test_dataloader, device
        )
        collection_time = time.time() - collection_start
        print(f"Attention scores collected in {collection_time:.2f} seconds")
        
        # Save attention scores
        save_attention_scores(pretrained_attention_scores, output_dirs["pretrained"])
        print(f"Saved pretrained model attention scores to {output_dirs['pretrained']}")
        
        # Clear model from GPU
        del pretrained_model
        del trainer_pretrained
        torch.cuda.empty_cache()
    
    # ======== PART 3: FINETUNED MODEL ========
    if args.finetuned:
        # Check if model already exists
        checkpoint_path = Path(full_checkpoint_path)
        if checkpoint_path.exists():
            print(f"Loading saved finetuned model from {full_checkpoint_path}...")
            finetuned_model = AutoModelForSequenceClassification.from_pretrained(
                full_checkpoint_path, trust_remote_code=True
            )
            finetuned_model.to(device)
        else:
            print(f"Checkpoint not found at {full_checkpoint_path}")
            print("Run finetuning script first to get finetuned model")
            return
                    
        # Enable attention collection
        finetuned_model.config.output_attentions = True
        finetuned_model.config.return_dict = True

        args_finetuned = TrainingArguments(
            output_dir="./tmp_finetuned",
            per_device_eval_batch_size=args.batch_size
        )

        trainer = Trainer(
            finetuned_model,
            args_finetuned,
            tokenizer=tokenizer
        )
        
        # Get test dataloader
        print("Getting test dataloader...")
        test_dataloader = trainer.get_test_dataloader(tokenized_test)
        print(f"Test dataloader created with {len(test_dataloader)} batches")
        
        # Collect attention scores
        print("Starting attention score collection...")
        collection_start = time.time()
        finetuned_attention_scores = collect_attention_scores(
            finetuned_model, tokenizer, test_dataloader, device
        )
        collection_time = time.time() - collection_start
        print(f"Attention scores collected in {collection_time:.2f} seconds")
        
        # Save attention scores
        print(f"Saving attention scores to {output_dirs['finetuned']}...")
        save_start = time.time()
        save_attention_scores(finetuned_attention_scores, output_dirs["finetuned"])
        save_time = time.time() - save_start
        print(f"Attention scores saved in {save_time:.2f} seconds")
        print(f"Saved finetuned model attention scores to {output_dirs['finetuned']}")

if __name__ == "__main__":
    main()