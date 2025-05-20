import gc
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
from anndata import AnnData
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import Vocab
from torchtext._torchtext import Vocab as VocabPybind

import wandb
from scipy.sparse import issparse
 
import scgpt
sys.path.insert(0, "../")
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.loss import masked_mse_loss, masked_relative_error, criterion_neg_log_bernoulli
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from torch.nn.utils import clip_grad_norm_
import pickle
torch.cuda.empty_cache()

class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def initialize_weights_normal(model: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    Initialize all model weights using a normal distribution N(mean, std).
    
    Args:
        model: PyTorch model
        mean: Mean of the normal distribution (default: 0.0)
        std: Standard deviation of the normal distribution (default: 0.02)
    """
    for module in model.modules():
        # Linear layers
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        
        # Embedding layers
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)  # gamma
            module.bias.data.zero_()       # beta
        
        # Conv layers
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        
        # Any other layer with parameters
        else:
            for param_name, param in module.named_parameters(recurse=False):
                if 'weight' in param_name:
                    # Apply normal initialization to any other weight parameters
                    param.data.normal_(mean=mean, std=std)
                elif 'bias' in param_name:
                    # Zero initialization for any bias parameters
                    param.data.zero_()


def random_init_scgpt(model, mean=0.0, std=0.02):
    """
    Random initialization for scGPT model.
    
    Args:
        model: scGPT model
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    
    Returns:
        The randomly initialized model
    """
    print("Applying standardized random initialization to scGPT model...")
    initialize_weights_normal(model, mean=mean, std=std)
    return model


def verify_standardized_initialization(model: nn.Module, 
                                      normal_mean: float = 0.0, 
                                      normal_std: float = 0.02,
                                      tolerance: float = 0.01) -> Dict[str, Any]:
    """
    Verify that model weights have been initialized according to our standardized scheme:
    - Linear layers: weights ~ N(mean, std), biases = 0
    - Embedding layers: weights ~ N(mean, std)
    - LayerNorm: gamma = 1.0, beta = 0.0
    - All other weight parameters: ~ N(mean, std)
    
    Args:
        model: PyTorch model
        normal_mean: Expected mean for normal distribution (default: 0.0)
        normal_std: Expected std for normal distribution (default: 0.02)
        tolerance: Tolerance for mean/std checks
        
    Returns:
        Dictionary with verification results
    """
    results = {
        "all_layers_verified": True,
        "layers_checked": 0,
        "issues": [],
        "stats": {}
    }
    
    # Counters for different layer types
    layer_counts = {
        "linear": 0,
        "embedding": 0,
        "layernorm": 0,
        "other": 0
    }
    
    # Collect weights by layer type for statistical analysis
    layer_weights = {
        "linear": [],
        "embedding": [],
        "other": []
    }
    
    # List to track verification of each layer
    layer_verifications = []
    
    print(f"\nVerifying initialization against standardized scheme:")
    print(f"- Linear layers: weights ~ N({normal_mean}, {normal_std}), biases = 0")
    print(f"- Embedding layers: weights ~ N({normal_mean}, {normal_std})")
    print(f"- LayerNorm: gamma = 1.0, beta = 0.0")
    print(f"- Other weight parameters: ~ N({normal_mean}, {normal_std})")
    print("-" * 80)
    
    # Check each named module
    for name, module in model.named_modules():
        # Skip top-level container modules to avoid redundant parameter checking
        if len(name.split('.')) <= 1 and not isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            continue
            
        if isinstance(module, nn.Linear):
            # Check linear layer weights
            layer_counts["linear"] += 1
            weights = module.weight.detach().cpu().numpy().flatten()
            layer_weights["linear"].extend(weights)
            
            # Calculate statistics
            w_mean = np.mean(weights)
            w_std = np.std(weights)
            
            # Check if weights match expected normal distribution
            weights_correct = (abs(w_mean - normal_mean) < tolerance and 
                              abs(w_std - normal_std) < tolerance * 3)
            
            # Check if biases are zeros
            bias_correct = True
            if module.bias is not None:
                biases = module.bias.detach().cpu().numpy().flatten()
                bias_mean = np.mean(np.abs(biases))
                bias_correct = bias_mean < tolerance / 10
            
            layer_correct = weights_correct and bias_correct
            
            # Log verification result
            layer_verifications.append({
                "name": name,
                "type": "linear",
                "weights_correct": weights_correct,
                "bias_correct": bias_correct,
                "w_mean": w_mean,
                "w_std": w_std,
                "correct": layer_correct
            })
            
            if not layer_correct:
                results["issues"].append({
                    "layer": name,
                    "type": "linear",
                    "weights_mean": w_mean,
                    "weights_std": w_std,
                    "expected_mean": normal_mean,
                    "expected_std": normal_std,
                    "bias_correct": bias_correct
                })
                results["all_layers_verified"] = False
                
        elif isinstance(module, nn.Embedding):
            # Check embedding layer weights
            layer_counts["embedding"] += 1
            weights = module.weight.detach().cpu().numpy()
            
            # Exclude padding token if it exists
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                if module.padding_idx < weights.shape[0]:
                    # Create a mask to exclude padding token
                    mask = np.ones(weights.shape[0], dtype=bool)
                    mask[module.padding_idx] = False
                    weights = weights[mask].flatten()
                else:
                    weights = weights.flatten()
            else:
                weights = weights.flatten()
                
            layer_weights["embedding"].extend(weights)
            
            # Calculate statistics
            w_mean = np.mean(weights)
            w_std = np.std(weights)
            
            # Check if weights match expected normal distribution
            weights_correct = (abs(w_mean - normal_mean) < tolerance and 
                              abs(w_std - normal_std) < tolerance * 3)
            
            layer_verifications.append({
                "name": name,
                "type": "embedding",
                "weights_correct": weights_correct,
                "w_mean": w_mean,
                "w_std": w_std,
                "correct": weights_correct
            })
            
            if not weights_correct:
                results["issues"].append({
                    "layer": name,
                    "type": "embedding",
                    "weights_mean": w_mean,
                    "weights_std": w_std,
                    "expected_mean": normal_mean,
                    "expected_std": normal_std
                })
                results["all_layers_verified"] = False
                
        elif isinstance(module, nn.LayerNorm):
            # Check LayerNorm parameters
            layer_counts["layernorm"] += 1
            
            # LayerNorm should have gamma=1 and beta=0
            gamma = module.weight.detach().cpu().numpy().flatten()
            beta = module.bias.detach().cpu().numpy().flatten()
            
            gamma_correct = np.allclose(gamma, 1.0, atol=tolerance)
            beta_correct = np.allclose(beta, 0.0, atol=tolerance)
            
            layer_correct = gamma_correct and beta_correct
            
            layer_verifications.append({
                "name": name,
                "type": "layernorm",
                "gamma_correct": gamma_correct,
                "beta_correct": beta_correct,
                "gamma_mean": np.mean(gamma),
                "beta_mean": np.mean(beta),
                "correct": layer_correct
            })
            
            if not layer_correct:
                results["issues"].append({
                    "layer": name,
                    "type": "layernorm",
                    "gamma_mean": np.mean(gamma),
                    "gamma_expected": 1.0,
                    "beta_mean": np.mean(beta),
                    "beta_expected": 0.0
                })
                results["all_layers_verified"] = False
                
        elif len(list(module.parameters(recurse=False))) > 0:
            # Check parameters of other layer types
            for param_name, param in module.named_parameters(recurse=False):
                # Only check weight parameters, not biases
                if 'weight' in param_name:
                    layer_counts["other"] += 1
                    weights = param.detach().cpu().numpy().flatten()
                    layer_weights["other"].extend(weights)
                    
                    w_mean = np.mean(weights)
                    w_std = np.std(weights)
                    
                    # Check if weights match expected normal distribution
                    weights_correct = (abs(w_mean - normal_mean) < tolerance and 
                                      abs(w_std - normal_std) < tolerance * 3)
                    
                    full_name = f"{name}.{param_name}"
                    layer_verifications.append({
                        "name": full_name,
                        "type": "other",
                        "weights_correct": weights_correct,
                        "w_mean": w_mean,
                        "w_std": w_std,
                        "correct": weights_correct
                    })
                    
                    if not weights_correct:
                        results["issues"].append({
                            "layer": full_name,
                            "type": "other",
                            "weights_mean": w_mean,
                            "weights_std": w_std,
                            "expected_mean": normal_mean,
                            "expected_std": normal_std
                        })
                        results["all_layers_verified"] = False
                
                # Check biases
                elif 'bias' in param_name:
                    biases = param.detach().cpu().numpy().flatten()
                    bias_mean = np.mean(np.abs(biases))
                    bias_correct = bias_mean < tolerance / 10
                    
                    full_name = f"{name}.{param_name}"
                    if not bias_correct:
                        results["issues"].append({
                            "layer": full_name,
                            "type": "bias",
                            "bias_mean": bias_mean,
                            "expected": 0.0
                        })
                        results["all_layers_verified"] = False
        
        results["layers_checked"] += 1
    
    # Calculate overall statistics for each layer type
    for layer_type, weights in layer_weights.items():
        if weights:
            results["stats"][layer_type] = {
                "mean": np.mean(weights),
                "std": np.std(weights),
                "min": np.min(weights),
                "max": np.max(weights),
                "count": len(weights)
            }
    
    # Count how many layers passed verification
    correct_layers = sum(1 for v in layer_verifications if v["correct"])
    total_layers = len(layer_verifications)
    
    # Print verification results
    print(f"\nVerification Summary:")
    print(f"- Linear layers: {layer_counts['linear']}")
    print(f"- Embedding layers: {layer_counts['embedding']}")
    print(f"- LayerNorm layers: {layer_counts['layernorm']}")
    print(f"- Other parameter layers: {layer_counts['other']}")
    print(f"- Total layers checked: {results['layers_checked']}")
    print(f"- Layers correctly initialized: {correct_layers}/{total_layers}")
    
    print("\nWeight Statistics:")
    for layer_type, stats in results["stats"].items():
        print(f"- {layer_type.capitalize()} layers: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    
    # Plot weight distribution
    plt.figure(figsize=(12, 8))
    
    # Plot histograms for each layer type
    plt.subplot(2, 2, 1)
    for layer_type, weights in layer_weights.items():
        if weights:
            plt.hist(weights, bins=50, alpha=0.5, label=f'{layer_type.capitalize()}')
    plt.title('Weight Distribution by Layer Type')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot Q-Q plot for linear layer weights
    if layer_weights["linear"]:
        plt.subplot(2, 2, 2)
        from scipy import stats
        linear_sample = np.random.choice(layer_weights["linear"], 
                                         size=min(1000, len(layer_weights["linear"])))
        stats.probplot(linear_sample, dist="norm", plot=plt)
        plt.title('Q-Q Plot (Linear Layer Weights)')
    
    # Plot initialization correctness by layer type
    plt.subplot(2, 2, 3)
    layer_types = ["linear", "embedding", "layernorm", "other"]
    correct_by_type = {layer_type: 0 for layer_type in layer_types}
    total_by_type = {layer_type: 0 for layer_type in layer_types}
    
    for v in layer_verifications:
        if v["type"] in total_by_type:
            total_by_type[v["type"]] += 1
            if v["correct"]:
                correct_by_type[v["type"]] += 1
    
    # Calculate percentage correct for each type
    pct_correct = []
    for layer_type in layer_types:
        if total_by_type[layer_type] > 0:
            pct = 100 * correct_by_type[layer_type] / total_by_type[layer_type]
            pct_correct.append(pct)
        else:
            pct_correct.append(0)
    
    plt.bar(layer_types, pct_correct)
    plt.title('Initialization Correctness by Layer Type')
    plt.xlabel('Layer Type')
    plt.ylabel('Percent Correct')
    plt.ylim(0, 105)
    
    # Add text labels
    for i, pct in enumerate(pct_correct):
        if total_by_type[layer_types[i]] > 0:
            plt.text(i, pct + 2, f"{pct:.1f}%", ha='center')
            plt.text(i, pct/2, f"{correct_by_type[layer_types[i]]}/{total_by_type[layer_types[i]]}", 
                    ha='center', color='white')
    
    # Overall correctness gauge
    plt.subplot(2, 2, 4)
    overall_pct = 100 * correct_layers / total_layers if total_layers > 0 else 0
    plt.pie([overall_pct, 100-overall_pct], 
            labels=[f'Correct ({correct_layers})', f'Issues ({total_layers-correct_layers})'],
            colors=['green', 'red'], autopct='%1.1f%%', 
            startangle=90)
    plt.axis('equal')
    plt.title('Overall Initialization Correctness')
    
    plt.tight_layout()
    plt.savefig('initialization_verification.png')
    print(f"\nVisualization saved to: initialization_verification.png")
    
    # Final verdict
    if results["all_layers_verified"]:
        print("\n✅ All layers are correctly initialized according to the standardized scheme!")
    else:
        print(f"\n⚠️ Found {len(results['issues'])} issues with initialization.")
        print("Here are the top issues:")
        for i, issue in enumerate(results["issues"][:5]):  # Show top 5 issues
            print(f"  {i+1}. {issue['layer']} ({issue['type']}): ", end="")
            if issue['type'] == 'layernorm':
                print(f"gamma={issue['gamma_mean']:.4f} (expected: 1.0), "
                      f"beta={issue['beta_mean']:.4f} (expected: 0.0)")
            elif issue['type'] == 'bias':
                print(f"bias_mean={issue['bias_mean']:.6f} (expected: 0.0)")
            else:
                print(f"mean={issue['weights_mean']:.6f} (expected: {normal_mean}), "
                      f"std={issue['weights_std']:.6f} (expected: {normal_std})")
        if len(results["issues"]) > 5:
            print(f"  ... and {len(results['issues'])-5} more issues.")
    
    return results


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }

    return train_data_pt, valid_data_pt


def analyze_attention_heads(model, data_loader, device, vocab, nlayers, nhead, pad_token, config):
    # settings for prediction
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = config.MVC  # Masked value prediction for cell embedding
    ECS = config.ecs_thres > 0  # Elastic cell similarity objective
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"
    ecs_threshold = config.ecs_thres
    dab_weight = config.dab_weight

    explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
    do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

    num_layers = nlayers 
    num_heads = nhead  

    model.to(device)
    model.eval()

    #dictionaries to store all examples and their scores for each head
    examples_scores_attention = {layer: {head: [] for head in range(num_heads)} for layer in range(num_layers)}

    for batch_num, batch_data in enumerate(data_loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)
        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])

        #output dictionary 
        output_dict = model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,
                    #generative_training = False,
                )
        input_tokens = [vocab.lookup_tokens(ids.tolist()) for ids in input_gene_ids]
        outputs = output_dict["cls_output"]
       
        # get attention scores:
        all_attentions = output_dict["attentions"]  # assuming the model was set with output_attentions=True

        #get labels
        batch_labels_list = celltype_labels.detach().cpu().numpy().tolist()
        #get expression
        input_values_list = input_values.detach().cpu().numpy().tolist()
        print(len(input_values_list))

        # for each layer and head...
        for layer in range(num_layers):
            for head in range(num_heads):
                attention_scores = all_attentions[layer][:, head, :, :]

                # tokens, attention matrices, labels, and input_values together
                for tokens, att_matrix, label, values in zip(input_tokens, attention_scores, batch_labels_list, input_values_list):
                    max_att_scores = att_matrix.max(dim=0)[0].detach().cpu().numpy()
                    # append a tuple with max attention scores, tokens, label, and the specific input_values
                    examples_scores_attention[layer][head].append((max_att_scores, tokens, label, values))
                    
    return examples_scores_attention



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/scratch/ssd004/scratch/mconsens/scGPT/examples/dev_ms-Apr01-12-07",
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--task_name",
        default="classification",
        type=str,
        help="either pretrained or classification or random_init",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/scGPT/tutorials/test-scgpt-lrp.py",
        type=str,
        help="The full path to this file",
    )
    parser.add_argument(
        "--dataset",
        default="ms",
        type=str,
        help="The dataset name, ms or pancreas",
    )
    args = parser.parse_args()

    model_path = args.model_path
    task_name = args.task_name
    full_path = args.full_path
    dataset_name = args.dataset
   
    sc.set_figure_params(figsize=(6, 6))
    os.environ["KMP_WARNINGS"] = "off"
    warnings.filterwarnings('ignore')

    hyperparameter_defaults = dict(
        seed=2,
        dataset_name=dataset_name,
        do_train=True,
        load_model= model_path,
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
        include_zero_gene = True,
        freeze = False, #freeze
        DSBN = False,  # Domain-spec batchnorm
    )

    run = wandb.init(config=hyperparameter_defaults, project="scGPT", reinit=True, settings=wandb.Settings(start_method="fork"))
    config = wandb.config

    # set_seed(config.seed)


    # settings for input and preprocessing
    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    mask_ratio = config.mask_ratio
    mask_value = "auto"  # for masked values, now it should always be auto

    include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
    max_seq_len = 3001
    n_bins = config.n_bins

    # input/output representation
    input_style = "binned"  # "normed_raw", "log1p", or "binned"
    output_style = "binned"  # "normed_raw", "log1p", or "binned"

    # settings for training
    MLM = False  # whether to use masked language modeling, currently it is always on.
    CLS = True  # celltype classification objective
    ADV = False  # Adversarial training for batch correction
    CCE = False  # Contrastive cell embedding objective
    MVC = config.MVC  # Masked value prediction for cell embedding
    ECS = config.ecs_thres > 0  # Elastic cell similarity objective
    DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
    INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
    input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
    cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
    adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
    adv_D_delay_epochs = 0
    mvc_decoder_style = "inner product"
    ecs_threshold = config.ecs_thres
    dab_weight = config.dab_weight

    explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
    do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

    per_seq_batch_sample = False

    # settings for optimizer
    lr = config.lr  # TODO: test learning rate ratio between two tasks
    lr_ADV = 1e-3  # learning rate for discriminator, used when ADV is True
    batch_size = config.batch_size
    eval_batch_size = config.batch_size
    epochs = config.epochs
    schedule_interval = 1

    # settings for the model
    fast_transformer = config.fast_transformer
    fast_transformer_backend = "linear"  # "linear" or "flash"
    embsize = config.layer_size  # embedding dimension
    d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
    nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
    nhead = config.nhead  # number of heads in nn.MultiheadAttention
    dropout = config.dropout  # dropout probability

    # logging
    log_interval = 100  # iterations
    save_eval_interval = config.save_eval_interval  # epochs
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

    dataset_name = config.dataset_name
    save_dir = Path(f"./save/dev_{dataset_name}-{time.strftime('%b%d-%H-%M')}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save to {save_dir}")
    logger = scgpt.logger
    scgpt.utils.add_file_handler(logger, save_dir / "run.log")



    if dataset_name == "ms":
        data_dir = Path("/scratch/ssd004/scratch/mconsens/scGPT/data") 
        print("data_dir=", )
        adata = sc.read(data_dir / "c_data.h5ad")
        adata_test = sc.read(data_dir / "filtered_ms_adata.h5ad")
        adata.obs["celltype"] = adata.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata_test.obs["celltype"] = adata_test.obs["Factor Value[inferred cell type - authors labels]"].astype("category")
        adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
        adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"          
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
        adata_test.var.set_index(adata.var["Gene Symbol"], inplace=True)
        
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
                    
    load_model = config.load_model         
    if load_model is not None:
        ## Load weights from other fine-tuned model
        model_dir = model_path
        model_dir = Path(config.load_model)
        model_config_file = model_dir / "args.json"
        #model_file = model_dir / "best_model.pt"
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
    adata=adata_test
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
        output_attentions=True,
        use_batch_labels=INPUT_BATCH_LABELS,
        num_batch_labels=num_batch_types,
        domain_spec_batchnorm=config.DSBN,
        input_emb_style=input_emb_style,
        n_input_bins=n_input_bins,
        cell_emb_style=cell_emb_style,
        mvc_decoder_style=mvc_decoder_style,
        ecs_threshold=ecs_threshold,
        explicit_zero_prob=explicit_zero_prob,
        use_fast_transformer=True, #fast_transformer,
        fast_transformer_backend=fast_transformer_backend,
        pre_norm=config.pre_norm,
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
    if task_name == 'random_init':
        model = random_init_scgpt(model)
        # Verify initialization
        results = verify_standardized_initialization(model)
        
        # Print results summary
        print(f"Initialization verification complete:")
        print(f"- Layers checked: {results['layers_checked']}")
        print(f"- All layers verified: {results['all_layers_verified']}")
        
        if not results['all_layers_verified']:
            print(f"- Found {len(results['anomalies'])} anomalies")
            for i, anomaly in enumerate(results['anomalies'][:5]):  # Show only first 5
                print(f"  Anomaly {i+1}: {anomaly}")
            
            if len(results['anomalies']) > 5:
                print(f"  ... and {len(results['anomalies'])-5} more anomalies.")

    pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    # Freeze all pre-decoder weights
    for name, para in model.named_parameters():
        print("-"*20)
        print(f"name: {name}")
        if config.freeze and "encoder" in name and "transformer_encoder" not in name:
        # if config.freeze and "encoder" in name:
            print(f"freezing weights for: {name}")
            para.requires_grad = False

    post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad).values())

    logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
    logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
    wandb.log(
            {
                "info/pre_freeze_param_count": pre_freeze_param_count,
                "info/post_freeze_param_count": post_freeze_param_count,
            },
    )

    model.to(device)
    wandb.watch(model)

    if ADV:
        discriminator = AdversarialDiscriminator(
            d_model=embsize,
            n_cls=num_batch_types,
        ).to(device)

    all_counts = (
        adata.layers[input_layer_key].A
        if issparse(adata.layers[input_layer_key])
        else adata.layers[input_layer_key]
    )

    celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
    celltypes_labels = np.array(celltypes_labels)

    batch_ids = adata.obs["batch_id"].tolist()
    batch_ids = np.array(batch_ids)

    tokenized_test = tokenize_and_pad_batch(
        #all_counts[:1],
        all_counts[:499],
        gene_ids,
        max_len=500,
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
        batch_size=eval_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), eval_batch_size // 2),
        pin_memory=True,
    )
    
    results = analyze_attention_heads(model, test_loader, device, vocab, nlayers, nhead, pad_token, config)


    # unpack results
    examples_scores_attention  = results
    full_path = '/scratch/ssd004/scratch/mconsens/scGPT/'


    # directory exists
    os.makedirs(f'{full_path}/attention/{dataset_name}/{task_name}/', exist_ok=True)
    #save scores in layer-indexed-files
    for layer in range(nlayers):
        #examples_scores_attention for the layer
        attention_filename = f'{full_path}/attention/{dataset_name}/{task_name}/examples_scores_attention_layer{layer}.p'
        with open(attention_filename, 'wb') as f:
            pickle.dump(examples_scores_attention[layer], f)


if __name__ == "__main__":
    main()
