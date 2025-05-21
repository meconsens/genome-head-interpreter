import gc
import json
import os
import copy
import pickle
import shutil
import time
import traceback
import warnings
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import torch
from anndata import AnnData
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

from scipy.sparse import issparse
import sys
sys.path.insert(0, '/scratch/ssd004/scratch/mconsens/scGPT')  # Path to your custom scGPT
import scgpt
# Import scGPT modules
from scgpt.model.model import TransformerModel, AdversarialDiscriminator
#from scgpt.model import TransformerModel
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from sklearn.metrics import roc_curve, auc

# HeadZeroer for scGPT
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


class SeqDataset(Dataset):
    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def apply_head_zeroing(model, layer_idx, head_idx):
    """
    Apply head zeroing to a specific attention head by inserting a HeadZeroer module
    after the attention output in the specified layer.
    """
    print(f"Applying head zeroing to layer {layer_idx}, head {head_idx}")
    
    # Ensure we're working with the correct structure
    if not hasattr(model, 'transformer_encoder') or not hasattr(model.transformer_encoder, 'layers'):
        raise AttributeError("Model doesn't have the expected transformer_encoder.layers structure")
        
    target_layer = model.transformer_encoder.layers[layer_idx]
    
    # Get dimensions
    if hasattr(model, 'nhead'):
        num_heads = model.nhead
        print(f"num_heads: {num_heads}")
    else:
        num_heads = 4  # Based on your config
        print(f"Using configured num_heads: {num_heads}")
    
    if hasattr(model, 'embsize'):
        hidden_size = model.embsize
        print(f"hidden_size: {hidden_size}")
    else:
        hidden_size = 128  # Based on your config 
        print(f"Using configured hidden_size: {hidden_size}")
    
    # Calculate head dimension
    head_dim = hidden_size // num_heads
    
    # Capture the original attention output module
    if hasattr(target_layer.self_attn, 'out_proj'):
        original_attention_output = target_layer.self_attn.out_proj
    else:
        raise AttributeError(f"self_attn in layer {layer_idx} doesn't have out_proj attribute")
    
    # Create a HeadZeroer module
    head_zeroer = HeadZeroer(head_idx, head_dim)
    
    # Create a combined module that applies both the original processing and our zeroing
    class CombinedModule(torch.nn.Module):
        def __init__(self, original_module, zeroer):
            super().__init__()
            self.original_module = original_module
            self.zeroer = zeroer
            
        def forward(self, x):
            # First run the original module
            outputs = self.original_module(x)
            # Then apply our zeroer
            return self.zeroer(outputs)
    
    # Replace the original module with our combined one
    target_layer.self_attn.out_proj = CombinedModule(original_attention_output, head_zeroer)
    print(f"✓ Inserted HeadZeroer after attention output in layer {layer_idx} for head {head_idx}")
    
    return model


# Apply head zeroing to multiple heads
def apply_multiple_head_zeroing(model, heads_to_ablate):
    """Apply head zeroing to multiple attention heads."""
    # Make a deep copy of the model to avoid modifying the original
    ablated_model = copy.deepcopy(model)
    
    # Check how many layers are available
    if hasattr(ablated_model, 'transformer_encoder') and hasattr(ablated_model.transformer_encoder, 'layers'):
        num_layers = len(ablated_model.transformer_encoder.layers)
        print(f"Model has {num_layers} transformer layers")
    else:
        print("WARNING: Could not determine number of layers")
        num_layers = 12  # Assuming 12 layers based on your parameter dump
    
    for head_info in heads_to_ablate:
        layer_idx = head_info["layer"]
        head_idx = head_info["head"]
        
        # Skip if layer index is out of bounds
        if layer_idx >= num_layers:
            print(f"Skipping Layer{layer_idx}-Head{head_idx}: Layer index out of bounds (max: {num_layers-1})")
            continue
            
        print(f"Ablating {head_info.get('name', f'Layer{layer_idx}-Head{head_idx}')}")
        try:
            ablated_model = apply_head_zeroing(ablated_model, layer_idx, head_idx)
        except Exception as e:
            print(f"Failed to ablate Layer{layer_idx}-Head{head_idx}: {e}")
            traceback.print_exc()
    
    return ablated_model


# Evaluation function
def evaluate_model(model, data_loader, device, vocab, pad_token, num_types, input_batch_labels=False, config=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_results = []
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_data in data_loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)
            
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            
            output_dict = model(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if input_batch_labels or (config and config["DSBN"]) else None,
                CLS=True,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=False,
            )
            
            outputs = output_dict["cls_output"]
            #check if outputs has logits by printing a small section
            #print(f"Outputs shape: {outputs.shape}")
            #print(f"Outputs sample: {outputs[0, :5]}")
            #print(f"Outputs sample: {outputs[1]}")
            # Get predicted probabilities and labels
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            predictions = outputs.argmax(1).cpu().numpy()
            
            all_preds.extend(predictions)
            all_labels.extend(celltype_labels.cpu().numpy())
            all_probs.extend(probs)
            
            # Store results
            for i in range(len(input_gene_ids)):
                tokens = vocab.lookup_tokens(input_gene_ids[i].cpu().tolist())
                all_results.append({
                    'tokens': str(tokens),
                    'true_label': int(celltype_labels[i].cpu().numpy()),
                    'predicted': int(predictions[i]),
                    'prob_values': probs[i].tolist()
                })
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    
    # For multiclass, we calculate AUC for each class
    class_aucs = []
    for i in range(num_types):
        # Create binary labels for this class
        binary_labels = np.array(all_labels) == i
        if np.sum(binary_labels) > 0:  # Only calculate if we have positive examples
            # Get probabilities for this class
            class_probs = np.array([p[i] for p in all_probs])
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            class_aucs.append(auc(fpr, tpr))
    
    # Average AUC across all classes
    mean_auc = np.mean(class_aucs) if class_aucs else 0
    return {
        'accuracy': accuracy,
        'mean_auc': mean_auc,
        'class_aucs': class_aucs,
        'labels': all_labels,
        'probs': all_probs,
        'detailed_results': all_results,
    }


def plot_multi_class_roc(results_dict, output_path, model_name="baseline"):
    """Create a single ROC curve plot showing all classes for a specific model."""
    plt.figure(figsize=(12, 10))
    
    # Get data from the specified model
    labels = results_dict[model_name]['labels']
    probs = results_dict[model_name]['probs']
    
    # Get class names if available, otherwise use indices
    class_names = []
    try:
        if 'class_names' in results_dict:
            class_names = results_dict['class_names']
    except:
        pass
    
    # Use random colors for each class
    colors = plt.cm.get_cmap('tab20', len(probs[0]))
    
    # Get unique class labels
    class_labels = np.unique(labels)
    
    # Binarize labels for one-vs-rest calculation
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(labels, classes=class_labels)
    
    # Calculate macro-average AUC using sklearn's function
    from sklearn.metrics import roc_auc_score
    macro_auc = roc_auc_score(y_true_bin, np.array(probs), multi_class='ovr', average='macro')
    
    # Plot ROC curve for each class
    for class_idx in range(len(probs[0])):
        # Create binary labels for this class
        binary_labels = np.array(labels) == class_idx
        
        # Get probabilities for this class
        class_probs = np.array([p[class_idx] for p in probs])
        
        # Only calculate if we have positive examples
        if np.sum(binary_labels) > 0:
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            roc_auc = auc(fpr, tpr)
            
            # Use class name if available, otherwise use index
            class_label = f"Class {class_idx}"
            if class_idx < len(class_names):
                class_label = class_names[class_idx]
                
            plt.plot(
                fpr, tpr, 
                color=colors(class_idx), 
                label=f"{class_label} (AUC = {roc_auc:.2f})", 
                linewidth=2
            )
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
    
    # Add macro average to title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'Multi-class ROC Curves (One-vs-Rest)\nMacro-Average AUC = {macro_auc:.4f}', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-class ROC curves saved to {output_path}")
    plt.close()
    
    # Return the macro-average AUC
    return macro_auc

def compare_model_rocs(results_dict, output_path, important_head_title, unimportant_head_title):
    """Create separate multi-class ROC curve plots for each model."""
    # Create ROC curve for baseline model
    plot_multi_class_roc(results_dict, f"{output_path}/baseline_roc_curves.png", "baseline")
    
    # Create ROC curve for important heads model
    plot_multi_class_roc(results_dict, f"{output_path}/important_roc_curves.png", "important")
    
    # Create ROC curve for unimportant heads model
    plot_multi_class_roc(results_dict, f"{output_path}/unimportant_roc_curves.png", "unimportant")

def plot_accuracy_comparison(results_dict, output_path, important_head_title, unimportant_head_title):
    """Plot bar chart comparing accuracy across models."""
    plt.figure(figsize=(10, 6))
    
    models = ['baseline', 'important', 'unimportant']
    labels = ['Baseline', important_head_title, unimportant_head_title]
    accuracies = [results_dict[m]['accuracy'] for m in models]
    aucs = [results_dict[m]['mean_auc'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, aucs, width, label='Mean AUC')
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(accuracies):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    for i, v in enumerate(aucs):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accuracy comparison saved to {output_path}")
    plt.close()


def prepare_test_dataloader(adata, input_layer_key, vocab, max_seq_len, pad_token, pad_value, 
                           mask_ratio, mask_value, batch_size, include_zero_gene=True):
    """Prepare test data loader from AnnData object."""
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    
    genes = adata.var["gene_name"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)
    
    celltypes_labels = adata.obs["celltype_id"].tolist()
    celltypes_labels = np.array(celltypes_labels)
    
    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)
    
    tokenized_test = tokenize_and_pad_batch(
        #all_counts[:1],
        #all_counts,
        all_counts[:499],
        gene_ids,
        max_len=500,
        #max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    input_values_test = random_mask_value(
        tokenized_test["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    
    test_data_pt = {
        "gene_ids": tokenized_test["genes"],
        "values": input_values_test,
        "target_values": tokenized_test["values"],
        "batch_labels": torch.from_numpy(batch_ids).long(),
        "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    }
    
    test_loader = DataLoader(
        dataset=SeqDataset(test_data_pt),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size // 2),
        pin_memory=True,
    )
    
    return test_loader

def plot_performance_by_percentage(results_dict, percentages, output_path, important_head_title, unimportant_head_title):
    """
    Create a line plot showing how accuracy and AUC change with increasing percentage of heads ablated.
    """
    plt.figure(figsize=(12, 8))
    
    # Get baseline values
    baseline_acc = results_dict['baseline']['accuracy']
    baseline_auc = results_dict['baseline']['mean_auc']
    
    # Get values for each percentage of important heads
    important_accs = [results_dict[f'important_{p}']['accuracy'] for p in percentages]
    important_aucs = [results_dict[f'important_{p}']['mean_auc'] for p in percentages]
    
    # Get values for each percentage of unimportant heads
    unimportant_accs = [results_dict[f'unimportant_{p}']['accuracy'] for p in percentages]
    unimportant_aucs = [results_dict[f'unimportant_{p}']['mean_auc'] for p in percentages]
    
    # Create subplot for AUC
    plt.subplot(2, 1, 1)
    
    # Plot AUC lines
    plt.plot(percentages, important_aucs, 'o-', color='red', linewidth=2, 
             label=f'{important_head_title} Ablated')
    plt.plot(percentages, unimportant_aucs, 'o-', color='green', linewidth=2, 
             label=f'{unimportant_head_title} Ablated')
    
    # Add a horizontal line for the baseline
    plt.axhline(y=baseline_auc, color='blue', linestyle='--', 
                label=f'Baseline (AUC = {baseline_auc:.3f})')
    
    # Add labels and legend
    plt.ylabel('Mean AUC')
    plt.title('Impact of Ablation on AUC Performance')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Make the x-axis show percentages
    plt.xticks(percentages)
    
    # Create subplot for Accuracy
    plt.subplot(2, 1, 2)
    
    # Plot Accuracy lines
    plt.plot(percentages, important_accs, 'o-', color='red', linewidth=2, 
             label=f'{important_head_title} Ablated')
    plt.plot(percentages, unimportant_accs, 'o-', color='green', linewidth=2, 
             label=f'{unimportant_head_title} Ablated')
    
    # Add a horizontal line for the baseline
    plt.axhline(y=baseline_acc, color='blue', linestyle='--', 
                label=f'Baseline (Acc = {baseline_acc:.3f})')
    
    # Add labels and legend
    plt.xlabel('Percentage of Heads Ablated (%)')
    plt.ylabel('Accuracy')
    plt.title('Impact of Ablation on Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Make the x-axis show percentages
    plt.xticks(percentages)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Performance by percentage plot saved to {output_path}")
    plt.close()

def plot_all_roc_curves(results_dict, percentages, output_path, important_head_title, unimportant_head_title):
    """
    Create ROC curves comparing baseline and all percentages of important and unimportant heads.
    This uses one-vs-rest approach for multi-class classification.
    """
    plt.figure(figsize=(15, 10))
    
    # Get baseline data
    baseline_labels = np.array(results_dict['baseline']['labels'])
    baseline_probs = np.array(results_dict['baseline']['probs'])
    
    # Determine the number of classes from the predictions shape
    num_classes = baseline_probs.shape[1]
    
    # Get unique class labels from the actual labels
    unique_labels = np.unique(baseline_labels)
    n_unique = len(unique_labels)
    
    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
    
    # Calculate AUC for baseline
    binary_aucs = []
    
    # Create a custom binarized matrix that matches the shape of the predictions
    for class_idx in range(num_classes):
        binary_labels = (baseline_labels == class_idx).astype(int)
        class_probs = baseline_probs[:, class_idx]
        
        # Skip classes with no positive examples
        if np.sum(binary_labels) == 0:
            continue
            
        # Calculate ROC and AUC
        try:
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            class_auc = auc(fpr, tpr)
            binary_aucs.append(class_auc)
            
            # Plot baseline curve for this class (transparent)
            plt.plot(fpr, tpr, color='blue', alpha=0.2, linewidth=1)
        except Exception as e:
            print(f"Error calculating ROC for class {class_idx}: {e}")
            continue
    
    # Calculate macro-average AUC for baseline
    baseline_macro_auc = np.mean(binary_aucs) if binary_aucs else 0
    
    # Add baseline to legend
    plt.plot(
        [0], [0],  # Dummy point for legend
        color='blue', linewidth=3, 
        label=f'Baseline (Macro AUC = {baseline_macro_auc:.3f})'
    )
    
    # Colors for important heads (shades of red)
    important_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(percentages)))
    
    # Colors for unimportant heads (shades of green)
    unimportant_colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(percentages)))
    
    # Plot important heads at each percentage
    for i, percentage in enumerate(percentages):
        key = f'important_{percentage}'
        if key not in results_dict:
            continue
            
        labels = np.array(results_dict[key]['labels'])
        probs = np.array(results_dict[key]['probs'])
        
        # Calculate AUCs for each class
        imp_binary_aucs = []
        
        for class_idx in range(num_classes):
            binary_labels = (labels == class_idx).astype(int)
            class_probs = probs[:, class_idx]
            
            # Skip classes with no positive examples
            if np.sum(binary_labels) == 0:
                continue
                
            try:
                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                class_auc = auc(fpr, tpr)
                imp_binary_aucs.append(class_auc)
            except Exception as e:
                print(f"Error calculating ROC for important_{percentage}, class {class_idx}: {e}")
                continue
        
        # Calculate macro-average AUC
        macro_auc = np.mean(imp_binary_aucs) if imp_binary_aucs else 0
        
        # Add to legend
        plt.plot(
            [0], [0],  # Dummy point for legend
            color=important_colors[i], linewidth=2, 
            label=f'{important_head_title} {percentage}% (Macro AUC = {macro_auc:.3f})'
        )
    
    # Plot unimportant heads at each percentage
    for i, percentage in enumerate(percentages):
        key = f'unimportant_{percentage}'
        if key not in results_dict:
            continue
            
        labels = np.array(results_dict[key]['labels'])
        probs = np.array(results_dict[key]['probs'])
        
        # Calculate AUCs for each class
        unimp_binary_aucs = []
        
        for class_idx in range(num_classes):
            binary_labels = (labels == class_idx).astype(int)
            class_probs = probs[:, class_idx]
            
            # Skip classes with no positive examples
            if np.sum(binary_labels) == 0:
                continue
                
            try:
                fpr, tpr, _ = roc_curve(binary_labels, class_probs)
                class_auc = auc(fpr, tpr)
                unimp_binary_aucs.append(class_auc)
            except Exception as e:
                print(f"Error calculating ROC for unimportant_{percentage}, class {class_idx}: {e}")
                continue
        
        # Calculate macro-average AUC
        macro_auc = np.mean(unimp_binary_aucs) if unimp_binary_aucs else 0
        
        # Add to legend
        plt.plot(
            [0], [0],  # Dummy point for legend
            color=unimportant_colors[i], linewidth=2, linestyle='--',
            label=f'{unimportant_head_title} {percentage}% (Macro AUC = {macro_auc:.3f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves for Different Percentages of Ablated Heads\n(Macro-Average)', fontsize=16)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"All ROC curves saved to {output_path}")
    plt.close()
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--task_name",
        default="classification",
        type=str,
        help="Task name or identifier for the experiment",
    )
    parser.add_argument(
        "--dataset",
        default="ms",
        type=str,
        help="The dataset name (e.g., ms, pancreas)",
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        type=str,
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment_name",
        required=True,
        type=str,
        help="Name for the experiment (used in file naming)",
    )
    parser.add_argument(
        "--important_heads_file",
        required=True,
        type=str,
        help="Path to the Python file containing important heads definitions",
    )
    parser.add_argument(
        "--unimportant_heads_file",
        required=True,
        type=str,
        help="Path to the Python file containing unimportant heads definitions",
    )
    # Add percentages argument with default values
    parser.add_argument(
        "--percentages", 
        type=str, 
        default="5,10,20,30,40,50", 
        help="Comma-separated list of percentages of heads to ablate (e.g., 5,10,20,30,40,50)"
    )
    args = parser.parse_args()
    model_path = args.model_path
    task_name = args.task_name
    dataset_name = args.dataset
    results_dir = args.results_dir
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Load important and unimportant heads
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
    
    # Parse percentages from arguments
    percentages = [int(p) for p in args.percentages.split(',')]
    
    print(f"Loaded {len(important_heads)} important heads and {len(unimportant_heads)} unimportant heads")
    
    sc.set_figure_params(figsize=(6, 6))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    hyperparameter_defaults = dict(
        seed=2,
        dataset_name=dataset_name,
        do_train=True,
        load_model=model_path,
        mask_ratio=0.0,
        epochs=1,
        n_bins=51,
        MVC=False, # Masked value prediction for cell embedding
        ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
        dab_weight=0.0,
        lr=1e-4,
        batch_size=2,
        layer_size=128,
        nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhead=4,  # number of heads in nn.MultiheadAttention
        dropout=0.2,  # dropout probability
        schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
        save_eval_interval=5,
        fast_transformer=True,
        pre_norm=False,
        amp=True,  # Automatic Mixed Precision
        include_zero_gene=True,
        freeze=False, #freeze
        DSBN=False,  # Domain-spec batchnorm SET TO FALSE
    )

    config = hyperparameter_defaults

    set_seed(0)

    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = 0.0
    mask_value = "auto"  # for masked values, now it should always be auto

    include_zero_gene = True  # if True, include zero genes among hvgs in the training
    max_seq_len = 3001
    n_bins = 51

    # input/output representation
    input_style = "binned"  # "normed_raw", "log1p", or "binned"
    output_style = "binned"  # "normed_raw", "log1p", or "binned"

    # settings for training
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = config["MVC"]  # Masked value prediction for cell embedding
    ECS = config["ecs_thres"] > 0  # Elastic cell similarity objective
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"
    ecs_threshold = config["ecs_thres"]
    dab_weight = config["dab_weight"]

    explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
    do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

    per_seq_batch_sample = False

    # settings for optimizer
    lr = config["lr"]  # learning rate
    lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
    batch_size = config["batch_size"]
    eval_batch_size = config["batch_size"]
    epochs = config["epochs"]
    schedule_interval = 1

    # settings for the model
    fast_transformer = config["fast_transformer"]
    fast_transformer_backend = "linear"  # "linear" or "flash"
    embsize = config["layer_size"]  # embedding dimension
    d_hid = config["layer_size"]  # dimension of the feedforward network in TransformerEncoder
    nlayers = config["nlayers"]  # number of TransformerEncoderLayer in TransformerEncoder
    nhead = config["nhead"]  # number of heads in nn.MultiheadAttention
    dropout = config["dropout"]  # dropout probability

    # logging
    log_interval = 100  # iterations
    save_eval_interval = config["save_eval_interval"]  # epochs
    do_eval_scib_metrics = True

    # %% validate settings
    assert input_style in ["normed_raw", "log1p", "binned"]
    assert output_style in ["normed_raw", "log1p", "binned"]
    assert input_emb_style in ["category", "continuous", "scaling"]
    if input_style == "binned":
        if input_emb_style == "scaling":
            raise ValueError("input_emb_style `scaling` is not supported for binned input.")
    elif input_style == "log1p" or input_style == "normed_raw":
        if input_emb_style == "category":
            raise ValueError(
                "input_emb_style `category` is not supported for log1p or normed_raw input."
            )

    if input_emb_style == "category":
        mask_value = n_bins + 1
        pad_value = n_bins  # for padding gene expr values
        n_input_bins = n_bins + 2
    else:
        mask_value = -1
        pad_value = -2
        n_input_bins = n_bins

    if ADV and DAB:
        raise ValueError("ADV and DAB cannot be both True.")
    DAB_separate_optim = True if DAB > 1 else False

    dataset_name = config["dataset_name"]
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scgpt.logger
    scgpt.utils.add_file_handler(logger, save_dir / "run.log")

    if dataset_name == "ms":
        data_dir = Path("/scratch/ssd004/scratch/mconsens/scGPT/data") 
        print("data_dir=", data_dir)
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata.obs["batch_id"] = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"] = adata_test.obs["str_batch"] = "1"          
        adata.var.set_index(adata.var["gene_name"], inplace=True)
        adata_test.var.set_index(adata.var["gene_name"], inplace=True)
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="str_batch")
        # make the batch category column
        batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()

    ################## Pancreas Dataset ##################
    if dataset_name == "pancreas":
        data_dir = Path("/scratch/ssd004/scratch/mconsens/scGPT/data/pancreas")
        adata = sc.read(data_dir / "demo_train.h5ad")
        adata_test = sc.read(data_dir / "demo_test.h5ad")
        
        adata.obs["celltype"] = adata.obs["Celltype"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Celltype"].astype("category")
        
        adata.obs["batch_id"] = "0"
        adata_test.obs["batch_id"] = "1"    

        adata.obs["str_batch"] = adata.obs["batch_id"]
        adata_test.obs["str_batch"] = adata_test.obs["batch_id"]   
        
        adata.var.set_index(adata.var["Gene Symbol"], inplace=True)
        adata_test.var.set_index(adata_test.var["Gene Symbol"], inplace=True)
        
        adata.var["gene_name"] = adata.var["Gene Symbol"]
        adata_test.var["gene_name"] = adata_test.var["Gene Symbol"]
        
        data_is_raw = False
        filter_gene_by_counts = False
        adata_test_raw = adata_test.copy()
        adata = adata.concatenate(adata_test, batch_key="batch_id")
                
        # make the batch category column
        batch_id_labels = adata.obs["batch_id"].astype("category").cat.codes.values
        adata.obs["batch_id"] = batch_id_labels
        celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
        celltypes = adata.obs["celltype"].unique()
        num_types = len(np.unique(celltype_id_labels))
        id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
        adata.obs["celltype_id"] = celltype_id_labels
        adata.var["gene_name"] = adata.var.index.tolist()

    load_model = config["load_model"]       
    if load_model is not None:
        ## Load weights from other fine-tuned model
        model_dir = model_path
        model_dir = Path(config["load_model"])
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        shutil.copy(vocab_file, save_dir / "vocab.json")
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )
        adata = adata[:, adata.var["id_in_vocab"] >= 0]

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=filter_gene_by_counts,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )
    else:
        ## Load weights from other fine-tuned model
        model_dir = model_path
        model_dir = Path(config["load_model"])
        model_config_file = model_dir / "args.json"

        # model
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Loading from {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]

        preprocessor = Preprocessor(
            use_key="X",  # the key in adata.layers to use as raw data
            filter_gene_by_counts=filter_gene_by_counts,  # step 1
            filter_cell_by_counts=False,  # step 2
            normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
            result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
            log1p=data_is_raw,  # 4. whether to log1p the normalized data
            result_log1p_key="X_log1p",
            subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
            hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
            binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
            result_binned_key="X_binned",  # the key in adata.layers to store the binned data
        )

    adata_test = adata[adata.obs["str_batch"] == "1"]
    adata = adata[adata.obs["str_batch"] == "0"]

    preprocessor(adata, batch_key=None)
    preprocessor(adata_test, batch_key=None)

    input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
        "normed_raw": "X_normed",
        "log1p": "X_normed",
        "binned": "X_binned",
    }[input_style]
    #set to test
    adata = adata_test
    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )
    genes = adata.var["gene_name"].tolist()

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    num_batch_types = len(set(batch_ids))
    batch_ids = np.array(batch_ids)

    (
        train_data,
        valid_data,
        train_celltype_labels,
        valid_celltype_labels,
        train_batch_labels,
        valid_batch_labels,
    ) = train_test_split(
        all_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True
    )

    if load_model is None:
        vocab = Vocab(
            VocabPybind(genes + special_tokens, None)
        )  # bidirectional lookup [gene <-> int]
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)

    tokenized_train = tokenize_and_pad_batch(
        train_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,  # append <cls> token at the beginning
        include_zero_gene=include_zero_gene,
    )
    tokenized_valid = tokenize_and_pad_batch(
        valid_data,
        gene_ids,
        max_len=max_seq_len,
        vocab=vocab,
        pad_token=pad_token,
        pad_value=pad_value,
        append_cls=True,
        include_zero_gene=include_zero_gene,
    )
    logger.info(
        f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_train['genes'].shape[1]}"
    )
    logger.info(
        f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
        f"\n\t feature length: {tokenized_valid['genes'].shape[1]}"
    )

    # dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=3,
        n_cls=num_types if CLS else 1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=MVC,
        do_dab=DAB,
        output_attentions=False,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config["DSBN"],
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=False, #fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config["pre_norm"],
    )
    if load_model is not None:
        try:
            model.load_state_dict(torch.load(model_file))
            logger.info(f"Loading all model params from {model_file}")
        except:
            # only load params that are in the model and match the size
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                logger.info(f"Loading params {k} with shape {v.shape}")
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        if config["freeze"] and "encoder" in name and "transformer_encoder" not in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    
    model.to(device)

    # Prepare test data loader
    test_loader = prepare_test_dataloader(
        adata_test,
        input_layer_key,
        vocab,
        max_seq_len,
        pad_token,
        pad_value,
        mask_ratio,
        mask_value,
        eval_batch_size,
        include_zero_gene
    )
    
    # Define the important and unimportant head titles based on experiment name
    if "pancreas_ductal" in args.experiment_name.lower():
        important_head_title = 'Pancreas Ductal Cell Important Heads'
        unimportant_head_title = 'Pancreas Ductal Cell Unimportant Heads'
    elif "GOBP_cell_dev" in args.experiment_name.lower():
        important_head_title = 'GOBP Cell Development Important Heads'
        unimportant_head_title = 'GOBP Cell Development Unimportant Heads'
    else:
        important_head_title = 'Important Heads'
        unimportant_head_title = 'Unimportant Heads'
    
    # Print class distribution in test dataset
    logger.info("\n====== Class Distribution in Datasets ======")

    # For test data from test_loader
    test_labels = []
    for batch in test_loader:
        test_labels.extend(batch["celltype_labels"].numpy())

    unique_test_labels, test_counts = np.unique(test_labels, return_counts=True)
    test_percentages = test_counts / len(test_labels) * 100

    logger.info("Test set class distribution:")
    for label, count, percentage in zip(unique_test_labels, test_counts, test_percentages):
        logger.info(f"  Class {label} ({id2type[label]}): {count} samples ({percentage:.2f}%)")
    
    # Get total number of heads in the model
    num_layers = nlayers
    num_heads = nhead
    total_heads = num_layers * num_heads
    logger.info(f"Model has {total_heads} total attention heads ({num_layers} layers × {num_heads} heads)")

    # Calculate number of heads for each percentage
    heads_per_percentage = {p: int(round(p * total_heads / 100)) for p in percentages}
    logger.info(f"Will ablate these numbers of heads: {heads_per_percentage}")

    # Dictionary to store all results for comparison
    results_dict = {}
    
    # 1. Evaluate the original model (baseline)
    logger.info("\n====== Evaluating Original Model (All Heads Intact) ======")
    
    baseline_results = evaluate_model(
        model, 
        test_loader, 
        device, 
        vocab, 
        pad_token, 
        num_types,
        INPUT_BATCH_LABELS,
        config
    )
    results_dict['baseline'] = baseline_results
    
    # Save baseline detailed results
    baseline_df = pd.DataFrame(baseline_results['detailed_results'])
    os.makedirs(results_dir, exist_ok=True)
    baseline_df.to_csv(f"{results_dir}/{args.experiment_name}_baseline_results.tsv", sep='\t', index=False)
    
    logger.info(f"Baseline model - Accuracy: {baseline_results['accuracy']:.4f}, Mean AUC: {baseline_results['mean_auc']:.4f}")
    
    # 2. Ablate important heads at each percentage
    for percentage in percentages:
        num_heads_to_ablate = heads_per_percentage[percentage]
        
        # Take the first N heads from the important heads list
        heads_to_ablate = important_heads[:num_heads_to_ablate]
        
        logger.info(f"\n====== Ablating Top {percentage}% Important Heads ({num_heads_to_ablate} heads) ======")
        head_names = [head.get("name", f"Layer{head['layer']}-Head{head['head']}") for head in heads_to_ablate]
        logger.info(f"Ablating heads: {', '.join(head_names)}")
        
        important_model = apply_multiple_head_zeroing(model, heads_to_ablate)
        important_model.to(device)
        
        results = evaluate_model(
            important_model, 
            test_loader, 
            device, 
            vocab, 
            pad_token, 
            num_types,
            INPUT_BATCH_LABELS,
            config
        )
        results_dict[f'important_{percentage}'] = results
        
        # Save detailed results
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(f"{results_dir}/{args.experiment_name}_important_heads_{percentage}percent_results.tsv", sep='\t', index=False)
        
        logger.info(f"{important_head_title} {percentage}% ablated - Accuracy: {results['accuracy']:.4f}, Mean AUC: {results['mean_auc']:.4f}")
        
        # Clean up to save memory
        del important_model
        torch.cuda.empty_cache()
    
    # 3. Ablate unimportant heads at each percentage
    for percentage in percentages:
        num_heads_to_ablate = heads_per_percentage[percentage]
        
        # Take the first N heads from the unimportant heads list
        heads_to_ablate = unimportant_heads[:num_heads_to_ablate]
        
        logger.info(f"\n====== Ablating Top {percentage}% Unimportant Heads ({num_heads_to_ablate} heads) ======")
        head_names = [head.get("name", f"Layer{head['layer']}-Head{head['head']}") for head in heads_to_ablate]
        logger.info(f"Ablating heads: {', '.join(head_names)}")
        
        unimportant_model = apply_multiple_head_zeroing(model, heads_to_ablate)
        unimportant_model.to(device)
        
        results = evaluate_model(
            unimportant_model, 
            test_loader, 
            device, 
            vocab, 
            pad_token, 
            num_types,
            INPUT_BATCH_LABELS,
            config
        )
        results_dict[f'unimportant_{percentage}'] = results
        
        # Save detailed results
        df = pd.DataFrame(results['detailed_results'])
        df.to_csv(f"{results_dir}/{args.experiment_name}_unimportant_heads_{percentage}percent_results.tsv", sep='\t', index=False)
        
        logger.info(f"{unimportant_head_title} {percentage}% ablated - Accuracy: {results['accuracy']:.4f}, Mean AUC: {results['mean_auc']:.4f}")
        
        # Clean up to save memory
        del unimportant_model
        torch.cuda.empty_cache()
    
    # 4. Create summary table
    summary_data = {
        'Model': ['Baseline'],
        'Accuracy': [results_dict['baseline']['accuracy']],
        'Mean AUC': [results_dict['baseline']['mean_auc']]
    }
    
    for percentage in percentages:
        summary_data['Model'].append(f'{important_head_title} {percentage}% Ablated')
        summary_data['Accuracy'].append(results_dict[f'important_{percentage}']['accuracy'])
        summary_data['Mean AUC'].append(results_dict[f'important_{percentage}']['mean_auc'])
        
    for percentage in percentages:
        summary_data['Model'].append(f'{unimportant_head_title} {percentage}% Ablated')
        summary_data['Accuracy'].append(results_dict[f'unimportant_{percentage}']['accuracy'])
        summary_data['Mean AUC'].append(results_dict[f'unimportant_{percentage}']['mean_auc'])
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"{results_dir}/{args.experiment_name}_ablation_summary.tsv"
    summary_df.to_csv(summary_path, sep='\t', index=False)
    
    # Print summary table
    logger.info("\n====== ABLATION EXPERIMENT SUMMARY ======")
    logger.info(summary_df.to_string())
    
    # 5. Create visualizations
    logger.info("\n====== Creating Visualizations ======")
    
    # plot performance across percentages
    plot_performance_by_percentage(
        results_dict, 
        percentages, 
        f"{results_dir}/{args.experiment_name}_performance_by_percentage.png", 
        important_head_title, 
        unimportant_head_title
    )
    
    # Plot all ROC curves
    plot_all_roc_curves(
        results_dict, 
        percentages, 
        f"{results_dir}/{args.experiment_name}_all_roc_curves.png", 
        important_head_title, 
        unimportant_head_title
    )
    
    # Save the full results dictionary for further analysis
    with open(f"{results_dir}/{args.experiment_name}_full_results.pkl", 'wb') as f:
        pickle.dump(results_dict, f)
    
    logger.info(f"\nExperiment complete! Results saved to {results_dir}")


if __name__ == "__main__":
    main()