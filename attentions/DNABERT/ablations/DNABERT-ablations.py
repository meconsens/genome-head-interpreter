import torch
import torch.nn.functional as F
import sys
import random
import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, AutoTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
import argparse

# Path definitions for ENHANCER
# base_dir = '/scratch/ssd004/scratch/mconsens/DNABERT'
# model_dir = f'{base_dir}/examples/ft/6/full_enhancer'
# task = 'enhancer'
# dev_data_path = f'{base_dir}/examples/sample_data/ft/6/full_enhancer/dev.tsv'
# results_dir = f'{base_dir}/Transformer-Explainability/enhancer_ablation_results'


#Path definitions for TATA
# base_dir = '/scratch/ssd004/scratch/mconsens/DNABERT'
# model_dir = f'{base_dir}/examples/ft/6/TATA'  # Changed from enhancer to TATA
# task = 'TATA'  # Changed from enhancer to TATA
# dev_data_path = f'{base_dir}/examples/sample_data/ft/6/TATA/dev.tsv'
# results_dir = f'{base_dir}/Transformer-Explainability/TATA_ablation_results'

# HeadZeroer for DNABERT (similar to our previous implementation but adapted for BERT architecture)
class HeadZeroer(torch.nn.Module):
    """A module that zeroes out a specific head's contribution"""
    def __init__(self, head_idx, head_dim):
        super().__init__()
        self.head_idx = head_idx
        self.head_dim = head_dim
        self.start_idx = head_idx * head_dim
        self.end_idx = (head_idx + 1) * head_dim
        print(f"Initializing HeadZeroer for head {head_idx} (indices {self.start_idx}-{self.end_idx})")
        
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

# Dataset class for DNABERT (keeping only necessary functionality)
class DNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_seq_length=512):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Encode the sequence (already in kmer format)
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Remove the batch dimension
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sequence": sequence,
            "label": label
        }

# Load and prepare data (simplified to only keep real sequences)
def load_data(file_path, num_samples=1000):
    print(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, sep='\t')
    
    # Split by class
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]
    
    # Sample if there are enough examples, otherwise use all
    if len(pos_df) >= num_samples:
        pos_samples = pos_df.sample(num_samples, replace=False)
    else:
        print(f"Warning: Only {len(pos_df)} positive samples available")
        pos_samples = pos_df
        
    if len(neg_df) >= num_samples:
        neg_samples = neg_df.sample(num_samples, replace=False)
    else:
        print(f"Warning: Only {len(neg_df)} negative samples available")
        neg_samples = neg_df
    
    # Get sequences (already kmerized) and labels
    pos_kmer_sequences = pos_samples['sequence'].tolist()
    neg_kmer_sequences = neg_samples['sequence'].tolist()
    pos_labels = [1] * len(pos_kmer_sequences)
    neg_labels = [0] * len(neg_kmer_sequences)
    
    # Combine positive and negative examples
    all_sequences = pos_kmer_sequences + neg_kmer_sequences
    all_labels = pos_labels + neg_labels
    
    print(f"Loaded {len(all_sequences)} sequences ({sum(all_labels)} positive, {len(all_labels) - sum(all_labels)} negative)")
    return all_sequences, all_labels

# Function to apply head zeroing using our new approach
def apply_head_zeroing(model, layer_idx, head_idx):
    """
    Apply head zeroing to a specific attention head by inserting a HeadZeroer module
    after the attention output in the specified layer.
    """
    print(f"Applying head zeroing to layer {layer_idx}, head {head_idx}")
    
    # Ensure we're working with BERT architecture
    if not hasattr(model, 'bert') or not hasattr(model.bert, 'encoder') or not hasattr(model.bert.encoder, 'layer'):
        raise AttributeError("Model doesn't have the expected BERT structure")
        
    target_layer = model.bert.encoder.layer[layer_idx]
    
    # Get dimensions
    if hasattr(model.config, 'num_attention_heads'):
        num_heads = model.config.num_attention_heads
    else:
        num_heads = 12  # Default for BERT Base
        print(f"Warning: Could not determine num_heads, using default: {num_heads}")
    
    if hasattr(model.config, 'hidden_size'):
        hidden_size = model.config.hidden_size
    else:
        hidden_size = 768  # Default for BERT Base
        print(f"Warning: Could not determine hidden_size, using default: {hidden_size}")
    
    # Calculate head dimension
    head_dim = hidden_size // num_heads
    
    # Capture the original attention output module
    original_attention_output = target_layer.attention.output
    
    # Create a HeadZeroer module
    head_zeroer = HeadZeroer(head_idx, head_dim)
    
    # Create a combined module that applies both the original processing and our zeroing
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
            return self.zeroer(outputs)
    
    # Replace the original module with our combined one
    target_layer.attention.output = CombinedModule(original_attention_output, head_zeroer)
    print(f"✓ Inserted HeadZeroer after attention output in layer {layer_idx} for head {head_idx}")
    
    return model

# Apply head zeroing to multiple heads
def apply_multiple_head_zeroing(model, heads_to_ablate):
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

# Evaluation function
def evaluate_model(model, dataset, device, batch_size=32):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_results = []
    
    print("Evaluating model...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            sequences = batch['sequence']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predicted probabilities and labels
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
            
            # Store results
            for i in range(len(sequences)):
                all_results.append({
                    'sequence': sequences[i],
                    'true_label': int(labels[i]),
                    'predicted': int(predictions[i]),
                    'prob_positive': float(probs[i, 1].cpu().numpy())
                })
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
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

# Visualizations

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

def main():
    parser = argparse.ArgumentParser(description="Run DNABERT ablation experiments")
    parser.add_argument('--base_dir', type=str, default='/scratch/ssd004/scratch/mconsens/DNABERT', help='Base directory for DNABERT')
    parser.add_argument('--model_dir', type=str, default='/scratch/ssd004/scratch/mconsens/DNABERT/examples/ft/6/TATA', help='Directory containing the model')
    parser.add_argument('--task', type=str, default='TATA', help='Task name (e.g., TATA, enhancer)')
    parser.add_argument('--dev_data_path', type=str, default='/examples/sample_data/ft/6/TATA/dev.tsv', help='Path to the test data file')
    parser.add_argument('--results_dir', type=str, default='/scratch/ssd004/scratch/mconsens/DNABERT/Transformer-Explainability/genome-interpreter/ablation_results/TATA-kmer', help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default='TATA-kmer', help='Overall name for the experiment')
    parser.add_argument('--important_heads_file', type=str, default='/scratch/ssd004/scratch/mconsens/DNABERT/Transformer-Explainability/genome-interpreter/ablation_data/TATA-kmer/important_heads.py', help='Path to the important heads file')
    parser.add_argument('--unimportant_heads_file', type=str, default='/scratch/ssd004/scratch/mconsens/DNABERT/Transformer-Explainability/genome-interpreter/ablation_data/TATA-kmer/unimportant_heads.py', help='Path to the unimportant heads file')
    # Add percentages argument with default values
    parser.add_argument('--percentages', type=str, default='5,10,20,30,40,50', help='Comma-separated list of percentages of heads to ablate (e.g., 5,10,20,30,40,50)')
    args = parser.parse_args()

    base_dir = args.base_dir
    model_dir = args.model_dir if args.model_dir else f'{base_dir}/examples/ft/6/{args.task}'
    dev_data_path = args.dev_data_path if args.dev_data_path else f'{base_dir}/examples/sample_data/ft/6/{args.task}/dev.tsv'
    results_dir = args.results_dir if args.results_dir else f'{base_dir}/Transformer-Explainability/genome-interpreter/ablation_results/{args.task}'
    experiment_name = args.experiment_name
 
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

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Remove duplicates from unimportant heads
    unimportant_heads = [dict(t) for t in {tuple(sorted(d.items())) for d in unimportant_heads}]

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    original_model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)
    
    # Load data (simplified version without scrambled/random sequences)
    sequences, labels = load_data(dev_data_path, num_samples=1000)
    
    # Create dataset
    dataset = DNADataset(
        sequences,
        labels,
        tokenizer
    )

    # Parse percentages from arguments
    percentages = [int(p) for p in args.percentages.split(',')]
    
    # Get total number of heads in the model
    num_layers = len(original_model.bert.encoder.layer)
    num_heads = original_model.config.num_attention_heads
    total_heads = num_layers * num_heads
    print(f"Model has {total_heads} total attention heads ({num_layers} layers × {num_heads} heads)")

    # Calculate number of heads for each percentage
    heads_per_percentage = {p: int(round(p * total_heads / 100)) for p in percentages}
    print(f"Will ablate these numbers of heads: {heads_per_percentage}")

    #based on experiment name, pull out whether important heads are TATA-kmer, TSS, or GC associated for naming the plots
    if experiment_name == 'TATA-kmer':
        important_head_title = 'TATA-kmer Important Heads'
        unimportant_head_title = 'TATA-kmer Unimportant Heads'
    elif experiment_name == 'TSS':
        important_head_title = 'TSS Important Heads'
        unimportant_head_title = 'TSS Unimportant Heads'
    elif experiment_name == 'GC':
        important_head_title = 'GC Important Heads'
        unimportant_head_title = 'GC Unimportant Heads'
    
    # Dictionary to store all results for comparison
    results_dict = {}
    
    # 1. Evaluate the original model (baseline)
    print("\n====== Evaluating Original Model (All Heads Intact) ======")
    original_model.to(device)
    baseline_results = evaluate_model(original_model, dataset, device)
    results_dict['baseline'] = baseline_results
    
    # Save baseline detailed results
    baseline_df = pd.DataFrame(baseline_results['detailed_results'])
    baseline_df.to_csv(f"{results_dir}/{experiment_name}_baseline_results.tsv", sep='\t', index=False)
    
    print(f"Baseline model - Accuracy: {baseline_results['accuracy']:.4f}, ROC AUC: {baseline_results['roc_auc']:.4f}")
    
    # 2. Ablate important heads at each percentage
    for percentage in percentages:
        num_heads_to_ablate = heads_per_percentage[percentage]
        
        # Take the first N heads from the important heads list
        heads_to_ablate = important_heads[:num_heads_to_ablate]
        
        print(f"\n====== Ablating Top {percentage}% Important Heads ({num_heads_to_ablate} heads) ======")
        head_names = [head.get("name", f"Layer{head['layer']}-Head{head['head']}") for head in heads_to_ablate]
        print(f"Ablating heads: {', '.join(head_names)}")
        
        important_model = apply_multiple_head_zeroing(original_model, heads_to_ablate)
        important_model.to(device)
        results = evaluate_model(important_model, dataset, device)
        results_dict[f'important_{percentage}'] = results
        
        # Save detailed results
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(f"{results_dir}/{experiment_name}_important_heads_{percentage}percent_results.tsv", sep='\t', index=False)
        
        print(f"{important_head_title} {percentage}% ablated - Accuracy: {results['accuracy']:.4f}, ROC AUC: {results['roc_auc']:.4f}")
        
        # Clean up to save memory
        del important_model
        torch.cuda.empty_cache()
    
    # 3. Ablate unimportant heads at each percentage
    for percentage in percentages:
        num_heads_to_ablate = heads_per_percentage[percentage]
        
        # Take the first N heads from the unimportant heads list
        heads_to_ablate = unimportant_heads[:num_heads_to_ablate]
        
        print(f"\n====== Ablating Top {percentage}% Unimportant Heads ({num_heads_to_ablate} heads) ======")
        head_names = [head.get("name", f"Layer{head['layer']}-Head{head['head']}") for head in heads_to_ablate]
        print(f"Ablating heads: {', '.join(head_names)}")
        
        unimportant_model = apply_multiple_head_zeroing(original_model, heads_to_ablate)
        unimportant_model.to(device)
        results = evaluate_model(unimportant_model, dataset, device)
        results_dict[f'unimportant_{percentage}'] = results
        
        # Save detailed results
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(f"{results_dir}/{experiment_name}_unimportant_heads_{percentage}percent_results.tsv", sep='\t', index=False)
        
        print(f"{unimportant_head_title} {percentage}% ablated - Accuracy: {results['accuracy']:.4f}, ROC AUC: {results['roc_auc']:.4f}")
        
        # Clean up to save memory
        del unimportant_model
        torch.cuda.empty_cache()
    
    # 4. Create summary table
    summary_data = {
        'Model': ['Baseline'],
        'Accuracy': [results_dict['baseline']['accuracy']],
        'ROC AUC': [results_dict['baseline']['roc_auc']]
    }
    
    for percentage in percentages:
        summary_data['Model'].append(f'{important_head_title} {percentage}% Ablated')
        summary_data['Accuracy'].append(results_dict[f'important_{percentage}']['accuracy'])
        summary_data['ROC AUC'].append(results_dict[f'important_{percentage}']['roc_auc'])
        
    for percentage in percentages:
        summary_data['Model'].append(f'{unimportant_head_title} {percentage}% Ablated')
        summary_data['Accuracy'].append(results_dict[f'unimportant_{percentage}']['accuracy'])
        summary_data['ROC AUC'].append(results_dict[f'unimportant_{percentage}']['roc_auc'])
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{results_dir}/{experiment_name}_ablation_summary.tsv"
    summary_df.to_csv(summary_path, sep='\t', index=False)
    
    # Print summary table
    print("\n====== ABLATION EXPERIMENT SUMMARY ======")
    print(summary_df)
    
    # 5. Create visualizations
    print("\n====== Creating Visualizations ======")
    
    # Create new function to plot performance across percentages
    plot_performance_by_percentage(results_dict, percentages, f"{results_dir}/{experiment_name}_performance_by_percentage.png", 
                                  important_head_title, unimportant_head_title)
    
    # Plot ROC curves for each percentage
    plot_all_roc_curves(results_dict, percentages, f"{results_dir}/{experiment_name}_all_roc_curves.png", 
                      important_head_title, unimportant_head_title)
    
    print(f"\nExperiment complete! Results saved to {results_dir}")

# New function to plot performance metrics by percentage
def plot_performance_by_percentage(results_dict, percentages, output_path, important_head_title, unimportant_head_title):
    """
    Create a line plot showing how AUC changes with increasing percentage of heads ablated.
    """
    plt.figure(figsize=(12, 8))
    
    # Get baseline AUC
    baseline_auc = results_dict['baseline']['roc_auc']
    
    # Get AUC values for each percentage of important heads
    important_aucs = [results_dict[f'important_{p}']['roc_auc'] for p in percentages]
    
    # Get AUC values for each percentage of unimportant heads
    unimportant_aucs = [results_dict[f'unimportant_{p}']['roc_auc'] for p in percentages]
    
    # Plot the lines
    plt.plot(percentages, important_aucs, 'o-', color='red', linewidth=2, 
             label=f'{important_head_title} Ablated')
    plt.plot(percentages, unimportant_aucs, 'o-', color='green', linewidth=2, 
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

# New function to plot all ROC curves
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
        plt.plot(results_dict[key]['fpr'], results_dict[key]['tpr'], 
                 label=f'{important_head_title} {percentage}% (AUC = {results_dict[key]["roc_auc"]:.3f})', 
                 linewidth=2, color=important_colors[i], linestyle='-')
    
    # Plot ROC curves for unimportant heads
    for i, percentage in enumerate(percentages):
        key = f'unimportant_{percentage}'
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
    main()