import pandas as pd
import numpy as np
import argparse
from scipy import stats
import os
from config_functions import configure_DNABERT, configure_scgpt, configure_nucleotide_transformer

def check_head_specificity(df, attention_score_column, head_num, label_column='label'):
    """
    Check if a head activates more on a specific label.
    
    Args:
        df: DataFrame with sequences
        attention_score_column: Column name for head attention scores
        label_column: Column containing labels
        
    Returns:
        preferred_label: The label this head prefers (or None if non-specific)
        p_value: Statistical significance of the preference
    """
    # Get unique labels
    unique_labels = df[label_column].unique()
    
    if len(unique_labels) <= 1:
        return None, 1.0  # No preference if only one label
    
    # Store means for each label
    label_means = {}
    for label in unique_labels:
        # label represents the full list of labels so
        label = str(label).split(',')[0]
        label_subset = df[df[label_column] == label]
        label_means[label] = label_subset[attention_score_column].apply(lambda x: np.mean(x)).mean()

    
    # Find label with highest mean
    max_label_key = max(label_means, key=label_means.get)
    
    # Perform statistical test to determine if preference is significant
    # Use ANOVA if more than 2 labels, t-test if exactly 2
    p_value = 1.0
    if len(unique_labels) == 2:
        # Extract the two labels
        label1, label2 = unique_labels
        group1 = df[df[label_column] == label1][attention_score_column].apply(lambda x: np.mean(x))
        group2 = df[df[label_column] == label2][attention_score_column].apply(lambda x: np.mean(x))

        
        # Perform t-test
        _, p_value = stats.ttest_ind(group1, group2, equal_var=False)
    else:
        # Prepare data for ANOVA
        groups = []
        for label in unique_labels:
            group = df[df[label_column] == label][attention_score_column].apply(lambda x: np.mean(x)).values
            groups.append(group)
            
        # Perform ANOVA
        _, p_value = stats.f_oneway(*groups)
    
    #bonferroni correct the p-value for the total number of heads in the model
    p_value = p_value * head_num
    #check for significant p-value
    if p_value > 0.05:
        max_label_key = None

    # Return the label key (which is now hashable) and p-value
    return max_label_key, p_value

def run_label_specific_correlations(df, attention_score_columns, bio_feature_columns, seq_lengths, head_num, label_column='label'):
    """
    Calculate correlations for heads across each label subset separately.
    
    Args:
        df: DataFrame with sequences
        attention_score_columns: List of column names for head attention scores
        bio_feature_columns: List of column names for biological features
        seq_lengths: List or integer of sequence lengths
        head_num: Number of heads in the model
        label_column: Column containing labels
        
    Returns:
        label_coef_dict: Dictionary mapping (head, label) pairs to correlation coefficients
        head_specificity: Dictionary mapping heads to their preferred labels
    """
    # Dictionary to store correlations by (head, label) pairs
    label_coef_dict = {}
    # Still track head specificity for reference
    head_specificity = {}
    
    # Extract simple labels
    df['simple_label'] = df[label_column].apply(lambda x: str(x).split(',')[0])
    
    # Get unique labels
    unique_labels = df['simple_label'].unique()
    
    for layer_head in attention_score_columns:
        # Check if this head has label preference (for reference)
        preferred_label, p_value = check_head_specificity(df, layer_head, head_num, label_column)
        head_specificity[layer_head] = preferred_label if preferred_label is not None else "non-specific"
        
        print(f"Calculating correlations for head {layer_head} (specificity: {head_specificity[layer_head]})")
        
        # For each label, calculate correlations
        for label in unique_labels:
            # Get subset of data for this label
            subset_df = df[df['simple_label'] == label]
            
            if subset_df.empty:
                print(f"No sequences found for label {label} in head {layer_head}")
                continue
                
            # Create a unique key for this head-label combination
            head_label_key = f"{layer_head}_{label}"
            label_coef_dict[head_label_key] = []
            
            X = None
            Y = None
            
            # Process sequences in this subset
            for index, seq in subset_df.iterrows():
                # Handle sequence length
                if isinstance(seq_lengths, list):
                    seq_len = seq_lengths[index]
                else:
                    seq_len = seq_lengths
                
                # Reshape features
                x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((len(bio_feature_columns), seq_len)).T
                y = seq[layer_head]
                
                if X is not None:
                    X = np.concatenate([X, x])
                    Y = np.concatenate([Y, y])
                else:
                    X = x
                    Y = y
            
            # Check for shape mismatches
            if len(X) != len(Y):
                print(f"WARNING: Shape mismatch for {head_label_key}: X has {len(X)} rows, Y has {len(Y)} elements")
                label_coef_dict[head_label_key] = [0] * len(bio_feature_columns)
                continue
            
            # Handle NaN values
            X = np.nan_to_num(X)
            
            # Calculate Spearman correlation for each feature
            for col_iter in range(X.shape[1]):
                result = stats.spearmanr(X[:, col_iter], Y)
                if not np.isnan(result.statistic):
                    label_coef_dict[head_label_key].append(result.statistic)
                else:
                    label_coef_dict[head_label_key].append(0)
    
    return label_coef_dict, head_specificity

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
    #get the number of heads
    print(f'Number of Heads: {len(attention_score_columns)}')
    head_num = len(attention_score_columns)
    print(f'Bio Feature Columns: {bio_feature_columns}')
    print(f'Sequence Length: {seq_length}')
    print(f'Label Column: {args.label_column}')

    # Run analysis with specificity 
    label_coef_dict, head_specificity = run_label_specific_correlations(
        new_df, 
        attention_score_columns, 
        bio_feature_columns, 
        seq_length,
        head_num,
        args.label_column,
    )
    # Create a more complex dataframe structure
    # First, gather all unique labels
    all_labels = new_df['simple_label'].unique()

    # Create a multi-level dataframe
    coef_dfs = {}
    for label in all_labels:
        # Filter for just this label's correlations
        label_keys = [k for k in label_coef_dict.keys() if k.endswith(f"_{label}")]
        if not label_keys:
            continue
            
        # Extract head names without the label suffix
        head_names = [k.rsplit('_', 1)[0] for k in label_keys]
        
        # Create dataframe for this label
        label_data = {head: label_coef_dict[f"{head}_{label}"] for head, key in zip(head_names, label_keys)}
        coef_dfs[label] = pd.DataFrame.from_dict(label_data, orient="index", columns=bio_feature_columns)
        
        # Add specificity information
        coef_dfs[label]['specificity'] = pd.Series({head: head_specificity[head] for head in head_names})
        
        # Save each label's results separately
        coef_dfs[label].to_csv(f'{full_path}/data/coef/{args.model_name}/{args.model_subtype}_label_{label}_headcorr.csv')

    print(f"Results saved to: {full_path}/data/coef/{args.model_name}/{args.model_subtype}_label_*_headcorr.csv")

if __name__ == "__main__":
    main()