import pandas as pd
import numpy as np
import argparse
import os
import json

def calculate_feature_specific_thresholds(real_df, distribution_folder):
    """
    Calculate thresholds for feature-specific z-scores using random distribution.
    
    Args:
        real_df: DataFrame with correlation coefficients
        distribution_folder: Folder containing random coefficient distributions
        
    Returns:
        z_scores: DataFrame with z-scores
        feature_stats: Dictionary with feature statistics
    """
    # Look for centered coefficient files
    files = [f for f in os.listdir(distribution_folder) if f.startswith('centered_coef_') and f.endswith('.csv')]
    if not files:
        # Fallback to original coefficient files if no centered files found
        files = [f for f in os.listdir(distribution_folder) if f.startswith('coef_') and f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("No coefficient files found in the distribution folder.")
        print("Using original coefficient files for z-score calculation.")
    else:
        print(f"Using {len(files)} centered coefficient files for z-score calculation.")

    dfs = [pd.read_csv(os.path.join(distribution_folder, file), index_col=0) for file in files]
    dist_df = pd.concat(dfs, axis=1)
    dist_df = dist_df[[col for col in dist_df.columns if 'position' not in col]]

    feature_stats = {}
    for feature in real_df.columns:
        if feature in dist_df.columns:
            all_values = dist_df[feature].values.flatten()
            all_values = all_values[~np.isnan(all_values)]
            std_dev = np.std(all_values) if len(all_values) > 0 else 0
            feature_stats[feature] = {
                'std_dev': std_dev,
                'three_sigma': 3 * std_dev
            }

    z_scores_dict = {}
    for feature in real_df.columns:
        if feature in dist_df.columns:
            mean = dist_df[feature].mean(axis=1)
            std = dist_df[feature].std(axis=1).replace(0, np.nan)
            raw_z_scores = (real_df[feature] - mean) / std
            if feature in feature_stats:
                threshold = feature_stats[feature]['three_sigma']
                raw_z_scores.loc[raw_z_scores.abs() <= threshold] = 0
            z_scores_dict[feature] = raw_z_scores

    z_scores = pd.DataFrame(z_scores_dict, index=real_df.index)
    z_scores = z_scores.replace([np.inf, -np.inf, np.nan], 0)
    return z_scores, feature_stats

def get_label_mapping(model_name):
    """Return the appropriate label mapping based on the model name."""
    label_mappings = {
        "scgpt_ms": {
            0: 'PVALB-expressing interneuron', 
            1: 'SST-expressing interneuron', 
            2: 'SV2C-expressing interneuron', 
            3: 'VIP-expressing interneuron', 
            4: 'astrocyte', 
            5: 'cortical layer 2-3 excitatory neuron A', 
            6: 'cortical layer 2-3 excitatory neuron B', 
            7: 'cortical layer 4 excitatory neuron', 
            8: 'cortical layer 5-6 excitatory neuron', 
            9: 'endothelial cell', 
            10: 'microglial cell', 
            11: 'mixed excitatory neuron', 
            12: 'mixed glial cell', 
            13: 'oligodendrocyte A', 
            14: 'oligodendrocyte C', 
            15: 'oligodendrocyte precursor cell', 
            16: 'phagocyte', 
            17: 'pyramidal neuron'
        },
        "scgpt_pancreas": {
            0: 'MHC class II', 
            1: 'PP', 
            2: 'PSC', 
            3: 'acinar', 
            4: 'alpha', 
            5: 'beta', 
            6: 'delta', 
            7: 'ductal', 
            8: 'endothelial', 
            9: 'epsilon', 
            10: 'macrophage', 
            11: 'mast', 
            12: 'schwann', 
            13: 't_cell'
        },
        "DNABERT_TATA": {
            0: 'non-TATA', 
            1: 'TATA'
        },
        "NT_TATA": {
            0: 'non-TATA', 
            1: 'TATA'
        },
        "DNABERT_enhancers": {
            0: 'non-enhancer', 
            1: 'enhancer'
        },
        "NT_enhancers": {
            0: 'non-enhancer', 
            1: 'enhancer'
        }
    }
    
    return label_mappings.get(model_name, {})

def format_json_with_labels(df, label_specific_dfs, bio_features, model_name):
    """
    Format the data as JSON with label-specific sentences for each head.
    
    Args:
        df: DataFrame with overall z-scores
        label_specific_dfs: Dictionary mapping labels to DataFrames with label-specific z-scores
        bio_features: List of biological feature column names
        model_name: Name of the model
        
    Returns:
        json_str: JSON string with formatted data
    """
    label_mapping = get_label_mapping(model_name)
    
    # Get unique labels from the label-specific DataFrames
    unique_labels = list(label_specific_dfs.keys())
    
    json_structure = {}
    for layer_head, series in df.iterrows():
        # Get features with non-zero z-scores
        features_only = {k: v for k, v in series.items() if k in bio_features}
        non_zero = {k: int(v) for k, v in features_only.items() if v != 0}
        sentences = [[feature if feature not in non_zero else [feature, non_zero[feature]] for feature in bio_features]]
        
        # Initialize the head's JSON structure
        json_structure[layer_head] = {
            "name": layer_head,
            "explanation": "The main thing this head does is find...",
            "sentences": sentences
        }
        
        # Add label-specific sentences for all labels
        for label in unique_labels:
            if layer_head in label_specific_dfs[label].index:
                # Get label-specific features
                label_series = label_specific_dfs[label].loc[layer_head]
                label_features = {k: v for k, v in label_series.items() if k in bio_features}
                label_non_zero = {k: int(v) for k, v in label_features.items() if v != 0}
                
                # Try to get the text representation of the label
                try:
                    label_key = int(float(label))
                    text_label = label_mapping.get(label_key, str(label))
                except (ValueError, TypeError):
                    text_label = str(label)
                
                # Create label-specific sentences
                label_sentences = [[feature if feature not in label_non_zero 
                                  else [feature, label_non_zero[feature]] 
                                  for feature in bio_features]]
                
                # Add to the head's JSON with label-specific key
                json_structure[layer_head][f"{text_label}_sentences"] = label_sentences
        
    return json.dumps(json_structure, indent=4)

def calculate_label_specific_z_scores(args, label_specific_coef_dfs, global_coef_df):
    """
    Calculate z-scores for label-specific correlations.
    
    Args:
        args: Command-line arguments
        label_specific_coef_dfs: Dictionary mapping labels to DataFrames with label-specific correlations
        global_coef_df: DataFrame with global correlations
        
    Returns:
        z_scores_by_label: Dictionary mapping labels to DataFrames with label-specific z-scores
        overall_z_scores: DataFrame with overall z-scores
        feature_stats: Dictionary with feature statistics
    """
    # Get distribution folder
    dist_folder = f"{args.full_path}/data/distributions/{args.model_name}/"
    
    # Calculate global z-scores
    overall_z_scores, feature_stats = calculate_feature_specific_thresholds(global_coef_df, dist_folder)
    
    # Calculate z-scores for each label
    z_scores_by_label = {}
    for label, label_df in label_specific_coef_dfs.items():
        label_z_scores, _ = calculate_feature_specific_thresholds(label_df, dist_folder)
        z_scores_by_label[label] = label_z_scores
    
    return z_scores_by_label, overall_z_scores, feature_stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/mica/genome-head-interpreter/preprocessing/data/coef/", type=str)
    parser.add_argument("--full_path", default="/home/mica/genome-head-interpreter/preprocessing/", type=str)
    parser.add_argument("--model_name", default="DNABERT_TATA", type=str)
    parser.add_argument("--model_subtype", default="finetuned", type=str)
    parser.add_argument("--uncentered", action="store_true", help="Use uncentered correlation results")
    args = parser.parse_args()

    # Define paths
    if args.uncentered:
        model_coef_path = f"{args.data_path}{args.model_name}-UNCENTERED"
        prefix = ""
        global_coef_path = f"{model_coef_path}/{args.model_subtype}_global_headcorr.csv"
    else:
        model_coef_path = f"{args.data_path}{args.model_name}"
        global_coef_path = f"{model_coef_path}/{args.model_subtype}_global_centered_headcorr.csv"
        prefix = "centered_"

    output_model_name = f"{args.model_name}-UNCENTERED" if args.uncentered else args.model_name

        
    global_df = pd.read_csv(global_coef_path, index_col=0)
    
    # Load label-specific correlation files
    label_specific_coef_dfs = {}
    
    # Look for centered label-specific files first
    label_files = [f for f in os.listdir(model_coef_path) if
               f.startswith(f"{args.model_subtype}_label_") and
               f"{prefix}headcorr.csv" in f]
    
    if not label_files:
        print("No label-specific correlation files found. Make sure to run the correlation analysis first.")
        return
    
    print(f"Found {len(label_files)} label-specific correlation files with prefix '{prefix}'")
    
    for label_file in label_files:
        # Extract label from filename
        if prefix:
            # Format: "{subtype}_label_{label}_centered_headcorr.csv"
            label = label_file.replace(f"{args.model_subtype}_label_", "").replace(f"_{prefix}headcorr.csv", "")
        else:
            # Format: "{subtype}_label_{label}_headcorr.csv"
            label = label_file.replace(f"{args.model_subtype}_label_", "").replace("_headcorr.csv", "")
            
        label_df = pd.read_csv(os.path.join(model_coef_path, label_file), index_col=0)
        label_specific_coef_dfs[label] = label_df
    
    # Get bio features (remove position columns)
    bio_features = [col for col in global_df.columns if 'position' not in col]
    
    # Calculate z-scores
    z_scores_by_label, overall_z_scores, feature_stats = calculate_label_specific_z_scores(args, label_specific_coef_dfs, global_df)
    # Create output directories
    os.makedirs(f"{args.full_path}/data/z_scores/{output_model_name}", exist_ok=True)
    os.makedirs(f"{args.full_path}/data/explanation_prompts/{output_model_name}", exist_ok=True)
    os.makedirs(f"{args.full_path}/data/z_scores/{output_model_name}/{args.model_subtype}", exist_ok=True)
    os.makedirs(f"{args.full_path}/data/z_scores/{output_model_name}/{args.model_subtype}/label_specific", exist_ok=True)

    # Save feature statistics
    feature_stats_df = pd.DataFrame.from_dict({k: v for k, v in feature_stats.items()})
    feature_stats_df.to_csv(f'{args.full_path}/data/z_scores/{output_model_name}/{args.model_subtype}_{prefix}feature_thresholds.csv')

    # Save global z-scores
    overall_z_scores.to_csv(f"{args.full_path}/data/z_scores/{output_model_name}/{args.model_subtype}_{prefix}z_scores.csv")

    # Save label-specific z-scores
    for label, z_scores in z_scores_by_label.items():
        z_scores.to_csv(f"{args.full_path}/data/z_scores/{output_model_name}/{args.model_subtype}/label_specific/{label}_{prefix}z_scores.csv")

    # Save JSON explanations
    json_out = format_json_with_labels(overall_z_scores, z_scores_by_label, bio_features, args.model_name)
    with open(f"{args.full_path}/data/explanation_prompts/{output_model_name}/{args.model_subtype}_{prefix}centered.json", "w") as f:
        f.write(json_out)

    print(f"Z-score analysis complete for {args.model_name} ({args.model_subtype}) using {prefix}centered approach")
    print(f"Label-specific z-scores saved to: {args.full_path}/data/z_scores/{output_model_name}/{args.model_subtype}/label_specific/")
    print(f"JSON explanation prompts saved to: {args.full_path}/data/explanation_prompts/{output_model_name}/{args.model_subtype}_{prefix}centered.json")

if __name__ == "__main__":
    main()