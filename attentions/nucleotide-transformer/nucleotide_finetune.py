# Imports
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset, Dataset
from pathlib import Path
import pickle
from Bio import SeqIO
import os
import argparse

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

# Define the metric for the evaluation using the f1 score
def compute_metrics_f1_score(eval_pred):
    """Computes F1 score for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    return {'f1_score': f1_score(references, predictions)}

def collect_attention_scores(model, tokenizer, test_dataloader, device):
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

def main():
    parser = argparse.ArgumentParser(description='Finetune Nucleotide Transformer model')
    parser.add_argument('--dataset', type=str, choices=['CUSTOM', 'FAKE', 'ENHANCER'], 
                        required=True, help='Dataset type to use')
    parser.add_argument('--gpu', type=str, default="1", help='GPU device to use')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--load_checkpoint', action='store_true', help='Load from checkpoint if available')
    args = parser.parse_args()

    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set data paths based on dataset choice
    base_path = "/home/mica/nucleotide-transformer/data"
    
    if args.dataset == "CUSTOM":
        train_file = f"{base_path}/custom_TATA_train.fa"
        test_file = f"{base_path}/custom_TATA_test.fa"
    elif args.dataset == "FAKE":
        train_file = f"{base_path}/fake_TATA_train.fa"
        test_file = f"{base_path}/fake_TATA_test.fa"
    elif args.dataset == "ENHANCER":
        train_file = f"{base_path}/enhancer_train.fa"
        test_file = f"{base_path}/enhancer_test.fa"
    
    # Load datasets
    train_sequences, train_labels = parse_fasta(train_file)
    test_sequences, test_labels = parse_fasta(test_file)

    # Print dataset statistics
    print(f"Dataset: {args.dataset}")
    print(f"Number of labelled 1 samples in train: {train_labels.count(1)}")
    print(f"Number of labelled 0 samples in train: {train_labels.count(0)}")
    print(f"Number of labelled 1 samples in test: {test_labels.count(1)}")
    print(f"Number of labelled 0 samples in test: {test_labels.count(0)}")

    # Split training data to get validation set
    train_sequences, val_sequences, train_labels, val_labels = train_test_split(
        train_sequences, train_labels, test_size=0.1, random_state=42
    )

    print(f"Train samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")
    print(f"Test samples: {len(test_sequences)}")

    # Load model and tokenizer
    model_name = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set checkpoint directory path
    output_dir = f"/home/mica/NucleotideTransformer/{args.dataset}-nucleotide-transformer-finetuned-NucleotideTransformer"
    checkpoint_dir = Path(output_dir)
    
    # Check if loading from checkpoint
    if args.load_checkpoint and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        print("Loading saved finetuned model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            f"{checkpoint_dir}/checkpoint-1000", 
            trust_remote_code=True
        )
        # Log model architecture details
        num_layers = model.config.num_hidden_layers
        num_heads = model.config.num_attention_heads
        print(f"Model has {num_layers} layers and {num_heads} heads")
    else:
        # Initialize new model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, output_attentions=False, trust_remote_code=True
        )
    
    model = model.to(device)

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
            max_length=512,  # Adjust based on your sequences
            return_tensors="pt"
        )
        return outputs

    # Tokenize datasets
    tokenized_train = ds_train.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_val = ds_val.map(tokenize_function, batched=True, remove_columns=["data"])
    tokenized_test = ds_test.map(tokenize_function, batched=True, remove_columns=["data"])
    
    # Training arguments
    per_device_train_batch_size = 12
    per_device_eval_batch_size = 12
    
    training_args = TrainingArguments(
        output_dir,
        remove_unused_columns=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=100,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_score",
        label_names=["labels"],
        dataloader_drop_last=True,
        max_steps=5000,
    )

    # Initialize trainer
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_f1_score,
    )

    # Train or evaluate
    if not args.load_checkpoint or not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        print(f"Training model with {args.dataset} dataset...")
        train_results = trainer.train()
        trainer.save_model(checkpoint_dir)
        print("Training complete and model saved.")
    else:
        # Evaluate on test set if loading from checkpoint
        print("Evaluating model on test set...")
        test_results = trainer.evaluate(tokenized_test)
        print(f"Test results: {test_results}")

if __name__ == "__main__":
    main()