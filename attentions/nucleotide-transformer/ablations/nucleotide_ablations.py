import torch
import numpy as np
import argparse
import os
import random
import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import copy
from collections import Counter
from Bio import SeqIO

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, roc_curve, auc, precision_recall_curve


# Data preparation functions
def parse_fasta(fasta_file, max_samples_per_class=None):
    """Parse a FASTA file to extract sequences and labels based on format '|label' in description"""
    sequences = []
    labels = []
    
    # Collect sequences by class
    class_sequences = {0: [], 1: []}
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        # Extract label from the record description using split('|')[-1]
        label = int(record.description.split('|')[-1])
        class_sequences[label].append(sequence)
    
    # Sample from each class if max_samples_per_class is specified
    if max_samples_per_class:
        for label in [0, 1]:
            if len(class_sequences[label]) > max_samples_per_class:
                class_sequences[label] = random.sample(class_sequences[label], max_samples_per_class)
    
    # Convert to single lists
    for label in [0, 1]:
        for seq in class_sequences[label]:
            sequences.append(seq)
            labels.append(label)
    
    return sequences, labels


# Dataset class for Nucleotide Transformer
class DNASequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Tokenize the sequence
        encoding = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove batch dimension
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence": sequence,
            "label": torch.tensor(label, dtype=torch.long)
        }


# Head ablation with HeadZeroer
class HeadZeroer(torch.nn.Module):
    """A module that zeroes out a specific head's contribution"""
    def __init__(self, head_idx, head_dim):
        super().__init__()
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.start_idx = head_idx * head_dim
        self.end_idx = (head_idx + 1) * head_dim
        
    def forward(self, hidden_states):
        """
        Zero out the specific head's contribution in the hidden states.
        hidden_states: [batch_size, seq_len, hidden_size]
        """
        # Create a copy to avoid modifying the original
        modified = hidden_states.clone()
        
        # Zero out the part corresponding to this head
        modified[:, :, self.start_idx:self.end_idx] = 0.0
        
        return modified


def apply_head_zeroing(model, layer_idx, head_idx):
    """
    Apply head zeroing to a specific attention head by inserting a HeadZeroer module
    after the attention output in the specified layer.
    """
    print(f"Applying head zeroing to layer {layer_idx}, head {head_idx}")
    
    # Find transformer layers
    if hasattr(model, 'esm') and hasattr(model.esm, 'encoder') and hasattr(model.esm.encoder, 'layer'):
        target_layer = model.esm.encoder.layer[layer_idx]
        
        # Get dimensions
        if hasattr(model.config, 'num_attention_heads'):
            num_heads = model.config.num_attention_heads
        else:
            num_heads = 16
            print(f"Warning: Could not determine num_heads, using default: {num_heads}")
        
        if hasattr(model.config, 'hidden_size'):
            hidden_size = model.config.hidden_size
        else:
            hidden_size = target_layer.attention.self.query.weight.shape[0]
            print(f"Warning: Could not determine hidden_size, inferred: {hidden_size}")
        
        # Calculate head dimension
        head_dim = hidden_size // num_heads
        
        # Capture the original module
        original_attention_output = target_layer.attention.output
        
        # Create a new HeadZeroer module
        head_zeroer = HeadZeroer(head_idx, head_dim)
        
        # Create a combined module
        class CombinedModule(torch.nn.Module):
            def __init__(self, original_module, zeroer):
                super().__init__()
                self.original_module = original_module
                self.zeroer = zeroer
                
            def forward(self, hidden_states, input_tensor=None):
                # First run the original module
                if input_tensor is not None:
                    outputs = self.original_module(hidden_states, input_tensor)
                else:
                    outputs = self.original_module(hidden_states)
                    
                # Then apply our zeroer
                if isinstance(outputs, tuple):
                    modified_outputs = list(outputs)
                    modified_outputs[0] = self.zeroer(outputs[0])
                    return tuple(modified_outputs)
                else:
                    return self.zeroer(outputs)
        
        # Replace the original module with our combined one
        target_layer.attention.output = CombinedModule(original_attention_output, head_zeroer)
        print(f"✓ Inserted HeadZeroer after attention output in layer {layer_idx} for head {head_idx}")
    else:
        raise AttributeError("Model doesn't have the expected ESM structure")
    
    return model


def ablate_multiple_heads(model, heads_to_ablate):
    """Apply head zeroing to multiple attention heads."""
    # Make a deep copy of the model to avoid modifying the original
    ablated_model = copy.deepcopy(model)
    
    for head_info in heads_to_ablate:
        layer_idx = head_info["layer"]
        head_idx = head_info["head"]
        print(f"Ablating {head_info.get('name', f'Layer{layer_idx}-Head{head_idx}')}")
        try:
            ablated_model = apply_head_zeroing(ablated_model, layer_idx, head_idx)
        except Exception as e:
            print(f"Failed to ablate Layer{layer_idx}-Head{head_idx}: {e}")
    
    return ablated_model


# Evaluation functions
def evaluate_model(model, dataset, device, batch_size=8):
    """
    Evaluate model performance on a dataset.
    Returns detailed metrics and prediction results.
    """
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_labels = []
    all_probs = []
    all_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            sequences = batch['sequence']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            # Store detailed results
            for i in range(len(sequences)):
                all_results.append({
                    'sequence': sequences[i],
                    'true_label': int(labels[i].cpu().numpy()),
                    'predicted': int(predictions[i]),
                    'prob_positive': float(probs[i, 1].cpu().numpy())
                })
    
    # Calculate overall accuracy
    predictions = [1 if p > 0.5 else 0 for p in all_probs]
    accuracy = accuracy_score(all_labels, predictions)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    # Calculate precision-recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall,
        'labels': all_labels,
        'probs': all_probs,
        'detailed_results': all_results
    }


# Visualization functions
def plot_roc_curves(results_dict, output_path, important_head_title, unimportant_head_title):
    """
    Create ROC curves comparing baseline, important head ablation, and unimportant head ablation.
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curves
    plt.plot(results_dict['baseline']['fpr'], results_dict['baseline']['tpr'], 
             label=f'Baseline (AUC = {results_dict["baseline"]["roc_auc"]:.3f})', 
             linewidth=2, color='blue')
    
    plt.plot(results_dict['important']['fpr'], results_dict['important']['tpr'], 
             label=f'{important_head_title} Ablated (AUC = {results_dict["important"]["roc_auc"]:.3f})', 
             linewidth=2, color='red')
    
    plt.plot(results_dict['unimportant']['fpr'], results_dict['unimportant']['tpr'], 
             label=f'{unimportant_head_title} Ablated (AUC = {results_dict["unimportant"]["roc_auc"]:.3f})', 
             linewidth=2, color='green')
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {output_path}")
    plt.close()


def plot_precision_recall_curves(results_dict, output_path, important_head_title, unimportant_head_title):
    """
    Create precision-recall curves comparing baseline, important head ablation, 
    and unimportant head ablation.
    """
    plt.figure(figsize=(10, 8))
    
    # Calculate average precision for each model
    def calculate_avg_precision(precision, recall):
        # Calculate average precision (area under PR curve)
        return np.sum((recall[:-1] - recall[1:]) * precision[:-1])
    
    baseline_ap = calculate_avg_precision(results_dict['baseline']['precision'], 
                                         results_dict['baseline']['recall'])
    important_ap = calculate_avg_precision(results_dict['important']['precision'], 
                                          results_dict['important']['recall'])
    unimportant_ap = calculate_avg_precision(results_dict['unimportant']['precision'], 
                                            results_dict['unimportant']['recall'])
    
    # Plot precision-recall curves
    plt.plot(results_dict['baseline']['recall'], results_dict['baseline']['precision'], 
             label=f'Baseline (AP = {baseline_ap:.3f})', 
             linewidth=2, color='blue')
    
    plt.plot(results_dict['important']['recall'], results_dict['important']['precision'], 
             label=f'{important_head_title} Ablated (AP = {important_ap:.3f})', 
             linewidth=2, color='red')
    
    plt.plot(results_dict['unimportant']['recall'], results_dict['unimportant']['precision'], 
             label=f'{unimportant_head_title} Ablated (AP = {unimportant_ap:.3f})', 
             linewidth=2, color='green')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Precision-recall curves saved to {output_path}")
    plt.close()


def plot_probability_distributions(results_dict, output_path, important_head_title, unimportant_head_title):
    """
    Create histograms of predicted probabilities for each model, separated by true label.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    models = ['baseline', 'important', 'unimportant']
    titles = ['Baseline Model', f'{important_head_title} Ablated', f'{unimportant_head_title} Ablated']
    
    for i, model_name in enumerate(models):
        # Get predicted probabilities and true labels
        probs = np.array(results_dict[model_name]['probs'])
        labels = np.array(results_dict[model_name]['labels'])
        
        # Separate probabilities by true label
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        
        # Plot histogram for positive examples (true label = 1)
        axes[i, 0].hist(pos_probs, bins=20, alpha=0.7, color='green')
        axes[i, 0].set_title(f'{titles[i]} - Positive Examples')
        axes[i, 0].set_xlabel('Predicted Probability of Positive Class')
        axes[i, 0].set_ylabel('Count')
        axes[i, 0].set_xlim(0, 1)
        
        # Plot histogram for negative examples (true label = 0)
        axes[i, 1].hist(neg_probs, bins=20, alpha=0.7, color='red')
        axes[i, 1].set_title(f'{titles[i]} - Negative Examples')
        axes[i, 1].set_xlabel('Predicted Probability of Positive Class')
        axes[i, 1].set_ylabel('Count')
        axes[i, 1].set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Probability distributions saved to {output_path}")
    plt.close()


def run_ablation_experiments(model_path, data_file, results_dir, important_heads, unimportant_heads,
                        important_head_title, unimportant_head_title,
                        experiment_name="nucleotide_transformer_ablation", samples_per_class=1000, 
                        batch_size=32, gpu="0", percentages="5,10,20,30,40,50"):
    """
    Run ablation experiments and create ROC curves comparing baseline,
    important and unimportant head ablation at different percentages.
    
    Parameters:
        percentages: Comma-separated list of percentages of heads to ablate
    """
    # Set up device and random seeds
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Prepare data
    print(f"Loading data from {data_file}")
    
    # Load data from FASTA file
    sequences, labels = parse_fasta(data_file, max_samples_per_class=samples_per_class)
    print(f"Loaded {len(sequences)} sequences ({labels.count(1)} positive, {labels.count(0)} negative)")
    
    # Create dataset
    print("Creating dataset...")
    dataset = DNASequenceDataset(
        sequences,
        labels,
        tokenizer
    )
    
    # Parse percentages
    percentages = [int(p) for p in percentages.split(',')]
    
    # Get model information to calculate total number of heads
    temp_model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    num_layers = len(temp_model.esm.encoder.layer)
    num_heads = temp_model.config.num_attention_heads
    total_heads = num_layers * num_heads
    del temp_model  # Free memory
    torch.cuda.empty_cache()
    
    print(f"Model has {total_heads} total attention heads ({num_layers} layers × {num_heads} heads)")
    
    # Calculate number of heads for each percentage
    heads_per_percentage = {p: int(round(p * total_heads / 100)) for p in percentages}
    print(f"Will ablate these numbers of heads: {heads_per_percentage}")
    
    # Run experiments
    results_dict = {}
    
    # 1. Evaluate original model (baseline)
    print("\n===== Evaluating Original Model (No Ablation) =====")
    original_model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
    original_model.to(device)
    
    baseline_results = evaluate_model(original_model, dataset, device, batch_size=batch_size)
    results_dict['baseline'] = baseline_results
    
    # Save baseline detailed results
    baseline_df = pd.DataFrame(baseline_results['detailed_results'])
    baseline_df.to_csv(os.path.join(results_dir, f"{experiment_name}_baseline_results.tsv"), sep='\t', index=False)
    
    print(f"Baseline model - Accuracy: {baseline_results['accuracy']:.4f}, ROC AUC: {baseline_results['roc_auc']:.4f}")
    
    # Clean up to save memory
    del original_model
    torch.cuda.empty_cache()
    
    # 2. Ablate important heads at each percentage
    for percentage in percentages:
        num_heads_to_ablate = heads_per_percentage[percentage]
        
        # Take the first N heads from the important heads list
        heads_to_ablate = important_heads[:num_heads_to_ablate]
        
        print(f"\n===== Ablating Top {percentage}% Important Heads ({num_heads_to_ablate} heads) =====")
        head_names = [head.get("name", f"Layer{head['layer']}-Head{head['head']}") for head in heads_to_ablate]
        print(f"Ablating heads: {', '.join(head_names)}")
        
        # Load a fresh model for each experiment
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        model.to(device)
        
        try:
            model = ablate_multiple_heads(model, heads_to_ablate)
            results = evaluate_model(model, dataset, device, batch_size=batch_size)
            results_dict[f'important_{percentage}'] = results
            
            # Save detailed results
            df = pd.DataFrame(results['detailed_results'])
            df.to_csv(os.path.join(results_dir, f"{experiment_name}_important_heads_{percentage}percent_results.tsv"), sep='\t', index=False)
            
            print(f"{important_head_title} {percentage}% ablated - Accuracy: {results['accuracy']:.4f}, ROC AUC: {results['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error during important heads {percentage}% ablation: {e}")
        
        # Clean up to save memory
        del model
        torch.cuda.empty_cache()
    
    # 3. Ablate unimportant heads at each percentage
    for percentage in percentages:
        num_heads_to_ablate = heads_per_percentage[percentage]
        
        # Take the first N heads from the unimportant heads list
        heads_to_ablate = unimportant_heads[:num_heads_to_ablate]
        
        print(f"\n===== Ablating Top {percentage}% Unimportant Heads ({num_heads_to_ablate} heads) =====")
        head_names = [head.get("name", f"Layer{head['layer']}-Head{head['head']}") for head in heads_to_ablate]
        print(f"Ablating heads: {', '.join(head_names)}")
        
        # Load a fresh model for each experiment
        model = AutoModelForSequenceClassification.from_pretrained(model_path, trust_remote_code=True)
        model.to(device)
        
        try:
            model = ablate_multiple_heads(model, heads_to_ablate)
            results = evaluate_model(model, dataset, device, batch_size=batch_size)
            results_dict[f'unimportant_{percentage}'] = results
            
            # Save detailed results
            df = pd.DataFrame(results['detailed_results'])
            df.to_csv(os.path.join(results_dir, f"{experiment_name}_unimportant_heads_{percentage}percent_results.tsv"), sep='\t', index=False)
            
            print(f"{unimportant_head_title} {percentage}% ablated - Accuracy: {results['accuracy']:.4f}, ROC AUC: {results['roc_auc']:.4f}")
        except Exception as e:
            print(f"Error during unimportant heads {percentage}% ablation: {e}")
        
        # Clean up to save memory
        del model
        torch.cuda.empty_cache()
    
    # Create summary table
    summary_data = {
        'Model': ['Baseline'],
        'Accuracy': [results_dict['baseline']['accuracy']],
        'ROC AUC': [results_dict['baseline']['roc_auc']]
    }
    
    for percentage in percentages:
        if f'important_{percentage}' in results_dict:
            summary_data['Model'].append(f'{important_head_title} {percentage}% Ablated')
            summary_data['Accuracy'].append(results_dict[f'important_{percentage}']['accuracy'])
            summary_data['ROC AUC'].append(results_dict[f'important_{percentage}']['roc_auc'])
    
    for percentage in percentages:
        if f'unimportant_{percentage}' in results_dict:
            summary_data['Model'].append(f'{unimportant_head_title} {percentage}% Ablated')
            summary_data['Accuracy'].append(results_dict[f'unimportant_{percentage}']['accuracy'])
            summary_data['ROC AUC'].append(results_dict[f'unimportant_{percentage}']['roc_auc'])
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(results_dir, f"{experiment_name}_ablation_summary.tsv")
    summary_df.to_csv(summary_path, sep='\t', index=False)
    
    # Print summary
    print("\n===== ABLATION EXPERIMENT SUMMARY =====")
    print(summary_df)
    print(f"\nSummary results saved to {summary_path}")
    
    # Create visualizations
    print("\n===== Creating Visualizations =====")
    
    # Plot performance across percentages
    perf_path = os.path.join(results_dir, f"{experiment_name}_performance_by_percentage.png")
    plot_performance_by_percentage(results_dict, percentages, perf_path, important_head_title, unimportant_head_title)
    
    # Plot all ROC curves
    roc_path = os.path.join(results_dir, f"{experiment_name}_all_roc_curves.png")
    plot_all_roc_curves(results_dict, percentages, roc_path, important_head_title, unimportant_head_title)
    
    return results_dict, summary_df


# Add these two new visualization functions
def plot_performance_by_percentage(results_dict, percentages, output_path, important_head_title, unimportant_head_title):
    """
    Create a line plot showing how AUC changes with increasing percentage of heads ablated.
    """
    plt.figure(figsize=(12, 8))
    
    # Get baseline AUC
    baseline_auc = results_dict['baseline']['roc_auc']
    
    # Get AUC values for each percentage of important heads
    important_aucs = []
    for p in percentages:
        key = f'important_{p}'
        if key in results_dict:
            important_aucs.append(results_dict[key]['roc_auc'])
        else:
            important_aucs.append(None)
    
    # Get AUC values for each percentage of unimportant heads
    unimportant_aucs = []
    for p in percentages:
        key = f'unimportant_{p}'
        if key in results_dict:
            unimportant_aucs.append(results_dict[key]['roc_auc'])
        else:
            unimportant_aucs.append(None)
    
    # Filter out None values for plotting
    valid_imp_percentages = [p for i, p in enumerate(percentages) if important_aucs[i] is not None]
    valid_imp_aucs = [auc for auc in important_aucs if auc is not None]
    
    valid_unimp_percentages = [p for i, p in enumerate(percentages) if unimportant_aucs[i] is not None]
    valid_unimp_aucs = [auc for auc in unimportant_aucs if auc is not None]
    
    # Plot the lines
    if valid_imp_aucs:
        plt.plot(valid_imp_percentages, valid_imp_aucs, 'o-', color='red', linewidth=2, 
                label=f'{important_head_title} Ablated')
    
    if valid_unimp_aucs:
        plt.plot(valid_unimp_percentages, valid_unimp_aucs, 'o-', color='green', linewidth=2, 
                label=f'{unimportant_head_title} Ablated')
    
    # Add a horizontal line for the baseline
    plt.axhline(y=baseline_auc, color='blue', linestyle='--', 
                label=f'Baseline (AUC = {baseline_auc:.3f})')
    
    # Add labels and legend
    plt.xlabel('Percentage of Heads Ablated (%)')
    plt.ylabel('ROC AUC')
    plt.title('Impact of Ablation on Model Performance')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Make the x-axis show percentages
    plt.xticks(percentages)
    
    # Save the figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance by percentage plot saved to {output_path}")
    plt.close()


def plot_all_roc_curves(results_dict, percentages, output_path, important_head_title, unimportant_head_title):
    """
    Create ROC curves for baseline and all percentages of important and unimportant heads.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot baseline ROC curve
    plt.plot(results_dict['baseline']['fpr'], results_dict['baseline']['tpr'], 
             label=f'Baseline (AUC = {results_dict["baseline"]["roc_auc"]:.3f})', 
             linewidth=3, color='blue')
    
    # Colors for important heads (shades of red)
    important_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(percentages)))
    
    # Colors for unimportant heads (shades of green)
    unimportant_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(percentages)))
    
    # Plot ROC curves for important heads
    for i, percentage in enumerate(percentages):
        key = f'important_{percentage}'
        if key in results_dict:
            plt.plot(results_dict[key]['fpr'], results_dict[key]['tpr'], 
                    label=f'{important_head_title} {percentage}% (AUC = {results_dict[key]["roc_auc"]:.3f})', 
                    linewidth=2, color=important_colors[i], linestyle='-')
    
    # Plot ROC curves for unimportant heads
    for i, percentage in enumerate(percentages):
        key = f'unimportant_{percentage}'
        if key in results_dict:
            plt.plot(results_dict[key]['fpr'], results_dict[key]['tpr'], 
                    label=f'{unimportant_head_title} {percentage}% (AUC = {results_dict[key]["roc_auc"]:.3f})', 
                    linewidth=2, color=unimportant_colors[i], linestyle='--')
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Percentages of Ablated Heads')
    plt.legend(loc='lower right', fontsize='small')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All ROC curves saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ablation experiments with ROC curve analysis")
    
    parser.add_argument("--model_path", default='/home/mica/nucleotide-transformer/CUSTOM-nucleotide-transformer-finetuned-NucleotideTransformer/checkpoint-1000',
                        help="Path to the finetuned model")
    parser.add_argument("--data_file", default='/home/mica/nucleotide-transformer/data/custom_TATA_test.fa',
                        help="Path to the FASTA file with test data")
    parser.add_argument("--results_dir", type=str, default="/home/mica/nucleotide-transformer/genome-interpreter/ablation_results/custom_TATA/",
                        help="Directory to save results")
    parser.add_argument("--important_heads_file", default='/home/mica/nucleotide-transformer/genome-interpreter/ablation_data/custom_TATA/important_heads.py',
                        help="Py file containing list of important heads to ablate")
    parser.add_argument("--unimportant_heads_file", default='/home/mica/nucleotide-transformer/genome-interpreter/ablation_data/custom_TATA/unimportant_heads.py', 
                        help="Py file containing list of unimportant heads to ablate")
    parser.add_argument("--experiment_name", type=str, default="TSS_heads",
                        help="Name for this experiment (used in filenames)")
    parser.add_argument("--samples_per_class", type=int, default=1000,
                        help="Maximum number of samples to use per class")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU index to use")
    # Add percentages argument
    parser.add_argument("--percentages", type=str, default="5,10,20,30,40,50",
                        help="Comma-separated list of percentages of heads to ablate")
    
    args = parser.parse_args()
    
    # Load important and unimportant heads from python files
    important_heads = []
    unimportant_heads = []
    
    # Execute the Python files to load the head lists
    important_heads_globals = {}
    with open(args.important_heads_file, 'r') as f:
        exec(f.read(), important_heads_globals)
        important_heads = important_heads_globals.get('important_heads', [])
    
    unimportant_heads_globals = {}
    with open(args.unimportant_heads_file, 'r') as f:
        exec(f.read(), unimportant_heads_globals)
        unimportant_heads = unimportant_heads_globals.get('unimportant_heads', [])

    experiment_name = args.experiment_name

    if experiment_name == 'TATA-kmer':
        important_head_title = 'TATA-kmer Important Heads'
        unimportant_head_title = 'TATA-kmer Unimportant Heads'
    elif experiment_name == 'TSS':
        important_head_title = 'TSS Important Heads'
        unimportant_head_title = 'TSS Unimportant Heads'
    elif experiment_name == 'GC':
        important_head_title = 'GC Important Heads'
        unimportant_head_title = 'GC Unimportant Heads'
    
    # Run the experiments
    run_ablation_experiments(
        model_path=args.model_path,
        data_file=args.data_file,
        results_dir=args.results_dir,
        important_heads=important_heads,
        unimportant_heads=unimportant_heads,
        important_head_title=important_head_title,
        unimportant_head_title=unimportant_head_title,
        experiment_name=args.experiment_name,
        samples_per_class=args.samples_per_class,
        batch_size=args.batch_size,
        gpu=args.gpu,
        percentages=args.percentages
    )
