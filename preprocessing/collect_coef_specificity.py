import pandas as pd
import numpy as np
import argparse
from scipy import stats
import os
from config_functions import configure_DNABERT, configure_scgpt, configure_nucleotide_transformer

def run_centered_attention_correlations(df, attention_score_columns, bio_feature_columns, seq_lengths, label_column='label'):
    """
    Calculate correlations between centered attention scores and biological features
    for each label subset separately and for the whole dataset.
    
    Centered attention = token attention - mean head attention across all tokens
    
    Args:
        df: DataFrame with sequences
        attention_score_columns: List of column names for head attention scores
        bio_feature_columns: List of column names for biological features
        seq_lengths: List or integer of sequence lengths
        label_column: Column containing labels
        
    Returns:
        label_coef_dict: Dictionary mapping (head, label) pairs to correlation coefficients
        global_coef_dict: Dictionary mapping heads to correlation coefficients across the whole dataset
    """
    # Dictionary to store correlations by (head, label) pairs
    label_coef_dict = {}
    # Dictionary to store correlations across the entire dataset
    global_coef_dict = {}
    
    # Extract simple labels
    df['simple_label'] = df[label_column].apply(lambda x: str(x).split(',')[0])
    
    # Get unique labels
    unique_labels = df['simple_label'].unique()
    
    # First, precompute mean attention scores for each head across the entire dataset
    head_global_means = {}
    for layer_head in attention_score_columns:
        # Flatten all attention scores for this head across all sequences
        all_attention_values = []
        for _, seq in df.iterrows():
            all_attention_values.extend(seq[layer_head])
        
        # Calculate global mean for this head
        head_global_means[layer_head] = np.mean(all_attention_values)
        print(f"Global mean for {layer_head}: {head_global_means[layer_head]}")
    
    # Calculate correlations across the entire dataset
    for layer_head in attention_score_columns:
        print(f"Calculating global correlations for head {layer_head}")
        
        X = None
        Y_centered = None  # For centered attention scores
        
        # Process all sequences
        for index, seq in df.iterrows():
            # Handle sequence length
            if isinstance(seq_lengths, list):
                seq_len = seq_lengths[index]
            else:
                seq_len = seq_lengths
            
            # Reshape features
            x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((len(bio_feature_columns), seq_len)).T
            
            # Center the attention scores
            y_original = seq[layer_head]
            y_centered = np.array(y_original) - head_global_means[layer_head]
            
            if X is not None:
                X = np.concatenate([X, x])
                Y_centered = np.concatenate([Y_centered, y_centered])
            else:
                X = x
                Y_centered = y_centered
        
        # Check for shape mismatches
        if len(X) != len(Y_centered):
            print(f"WARNING: Global shape mismatch for {layer_head}: X has {len(X)} rows, Y has {len(Y_centered)} elements")
            global_coef_dict[layer_head] = [0] * len(bio_feature_columns)
            continue
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Calculate Spearman correlation for each feature using centered attention
        global_coef_dict[layer_head] = []
        for col_iter in range(X.shape[1]):
            result = stats.spearmanr(X[:, col_iter], Y_centered)
            if not np.isnan(result.statistic):
                global_coef_dict[layer_head].append(result.statistic)
            else:
                global_coef_dict[layer_head].append(0)
    
    # For each label, calculate correlations
    for label in unique_labels:
        # Get subset of data for this label
        subset_df = df[df['simple_label'] == label]
        
        if subset_df.empty:
            print(f"No sequences found for label {label}")
            continue
            
        print(f"Calculating correlations for label {label}")
        
        for layer_head in attention_score_columns:
            # Create a unique key for this head-label combination
            head_label_key = f"{layer_head}_{label}"
            label_coef_dict[head_label_key] = []
            
            X = None
            Y_centered = None
            
            # Process sequences in this subset
            for index, seq in subset_df.iterrows():
                # Handle sequence length
                if isinstance(seq_lengths, list):
                    seq_len = seq_lengths[index]
                else:
                    seq_len = seq_lengths
                
                # Reshape features
                x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((len(bio_feature_columns), seq_len)).T
                
                # Center the attention scores using the global mean
                y_original = seq[layer_head]
                y_centered = np.array(y_original) - head_global_means[layer_head]
                
                if X is not None:
                    X = np.concatenate([X, x])
                    Y_centered = np.concatenate([Y_centered, y_centered])
                else:
                    X = x
                    Y_centered = y_centered
            
            # Check for shape mismatches
            if len(X) != len(Y_centered):
                print(f"WARNING: Shape mismatch for {head_label_key}: X has {len(X)} rows, Y has {len(Y_centered)} elements")
                label_coef_dict[head_label_key] = [0] * len(bio_feature_columns)
                continue
            
            # Handle NaN values
            X = np.nan_to_num(X)
            
            # Calculate Spearman correlation for each feature using centered attention
            for col_iter in range(X.shape[1]):
                result = stats.spearmanr(X[:, col_iter], Y_centered)
                if not np.isnan(result.statistic):
                    label_coef_dict[head_label_key].append(result.statistic)
                else:
                    label_coef_dict[head_label_key].append(0)
    
    return label_coef_dict, global_coef_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/scores/",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/",
        type=str,
        help="The full path to the collect_coef.py file",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT_TATA",
        type=str,
        help="The model being explained",
    )
    parser.add_argument(
        "--model_subtype",
        default="finetuned",
        type=str,
        choices=["random_init", "random", "pretrained", "finetuned"],
        help="The model subtype being explained",
    )
    parser.add_argument(
        "--label_column",
        default="label",
        type=str,
        help="Column name containing sequence labels",
    )
    args = parser.parse_args()

    data_path = args.data_path
    full_path = args.full_path
    model = args.model_name
    
    # Make sure if model_subtype selected as "random", the model_name is DNABERT_TATA or DNABERT_enhancer
    if args.model_subtype == "random":
        assert args.model_name == "DNABERT_TATA" or args.model_name == "DNABERT_enhancers", "Model name should be DNABERT_TATA or DNABERT_enhancers"
    
    # Add model name to the data path
    data_path = f'{data_path}{args.model_name}/{args.model_name}_{args.model_subtype}_scores.csv'
    
    df = pd.read_csv(data_path, sep=';')
    
    # Verify label column exists
    if args.label_column not in df.columns:
        raise ValueError(f"Label column '{args.label_column}' not found in dataset")

    # Check for and drop _minmax columns
    minmax_columns = [col for col in df.columns if '_minmax' in col]
    if minmax_columns:
        print(f"Found {len(minmax_columns)} columns with '_minmax' suffix - dropping them")
        df = df.drop(columns=minmax_columns)
        print(f"Remaining columns: {len(df.columns)}")

    config_functions = {
        'DNABERT_TATA': configure_DNABERT,
        'DNABERT_enhancers': configure_DNABERT,
        'scgpt_ms': configure_scgpt,
        'scgpt_pancreas': configure_scgpt,
        'NT_TATA': configure_nucleotide_transformer,
        'NT_enhancers': configure_nucleotide_transformer
    }

    config_function = config_functions.get(args.model_name)
    
    config = config_function(df)

    new_df = config["transformed_df"] if "transformed_df" in config else df

    attention_score_columns = config["attention_score_columns"]
    bio_feature_columns = config["bio_feature_columns"]
    seq_length = config["seq_length"]

    print(f'Attention Score Columns: {attention_score_columns}')
    print(f'Number of Heads: {len(attention_score_columns)}')
    print(f'Bio Feature Columns: {bio_feature_columns}')
    print(f'Sequence Length: {seq_length}')
    print(f'Label Column: {args.label_column}')

    # Run analysis with centered attention correlations
    label_coef_dict, global_coef_dict = run_centered_attention_correlations(
        new_df, 
        attention_score_columns, 
        bio_feature_columns, 
        seq_length,
        args.label_column,
    )
    
    # Create a global correlation dataframe
    global_df = pd.DataFrame.from_dict(global_coef_dict, orient="index", columns=bio_feature_columns)
    global_df.to_csv(f'{full_path}/data/coef/{args.model_name}/{args.model_subtype}_global_centered_headcorr.csv')
    
    print(f"Global centered correlations saved to: {full_path}/data/coef/{args.model_name}/{args.model_subtype}_global_centered_headcorr.csv")
    
    # Create dataframes for each label
    all_labels = new_df['simple_label'].unique()
    for label in all_labels:
        # Filter for just this label's correlations
        label_keys = [k for k in label_coef_dict.keys() if k.endswith(f"_{label}")]
        if not label_keys:
            continue
            
        # Extract head names without the label suffix
        head_names = [k.rsplit('_', 1)[0] for k in label_keys]
        
        # Create dataframe for this label
        label_data = {head: label_coef_dict[f"{head}_{label}"] for head, key in zip(head_names, label_keys)}
        label_df = pd.DataFrame.from_dict(label_data, orient="index", columns=bio_feature_columns)
        
        # Save each label's results separately
        label_df.to_csv(f'{full_path}/data/coef/{args.model_name}/{args.model_subtype}_label_{label}_centered_headcorr.csv')
    
    print(f"Label-specific centered correlations saved to: {full_path}/data/coef/{args.model_name}/{args.model_subtype}_label_*_centered_headcorr.csv")

if __name__ == "__main__":
    main()