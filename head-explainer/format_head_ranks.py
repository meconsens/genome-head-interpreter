import pandas as pd
import numpy as np
import argparse
import os
import json

def calculate_feature_specific_thresholds(real_df, distribution_folder):
    # Find the coefficient distribution files
    print("Distribution folder:", distribution_folder)
    files = [f for f in os.listdir(distribution_folder) if f.startswith('coef_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No coefficient files found in the distribution folder.")
    
    print(f"Found {len(files)} distribution files in {distribution_folder}")
    
    # Load all distribution data
    dfs = [pd.read_csv(os.path.join(distribution_folder, file), index_col=0) for file in files]
    dist_df = pd.concat(dfs, axis=1)
    dist_df = dist_df[[col for col in dist_df.columns if 'position' not in col]]
    
    # Calculate 3-sigma threshold for each feature
    feature_stats = {}
    for feature in real_df.columns:
        if feature in dist_df.columns:
            # Get all values for this feature across all files
            all_values = dist_df[feature].values.flatten()
            all_values = all_values[~np.isnan(all_values)]  # Remove NaN values
            
            # Calculate statistics
            if len(all_values) > 0:
                std_dev = np.std(all_values)
                three_sigma = 3 * std_dev
                
                feature_stats[feature] = {
                    'std_dev': std_dev,
                    'three_sigma': three_sigma
                }
                
                print(f"Feature {feature}: std={std_dev:.4f}, 3σ threshold={three_sigma:.4f}")
            else:
                feature_stats[feature] = {
                    'std_dev': 0,
                    'three_sigma': 0
                }
    
    # Dictionary to store Z-scores
    z_scores_dict = {}
    z_scores_dict_no_cutoff = {}
    
    # Calculate Z-scores for each feature
    for feature in real_df.columns:
        if feature in dist_df.columns:
            mean = dist_df[feature].mean(axis=1)
            std = dist_df[feature].std(axis=1)
            std[std == 0] = np.nan  
            
            # Calculate raw Z-scores
            raw_z_scores = (real_df[feature] - mean) / std
            z_scores_dict_no_cutoff[feature] = raw_z_scores
            
            # Apply 3-sigma threshold (feature-specific)
            if feature in feature_stats:
                three_sigma_cutoff = feature_stats[feature]['three_sigma']
                z_scores_dict[feature] = raw_z_scores.copy()
                z_scores_dict[feature].loc[z_scores_dict[feature].abs() <= three_sigma_cutoff] = 0
    
    # Create DataFrames from dictionaries
    z_scores = pd.DataFrame(z_scores_dict, index=real_df.index)
    z_scores_no_cutoff = pd.DataFrame(z_scores_dict_no_cutoff, index=real_df.index)
    
    # Replace NaN and infinite values with 0
    z_scores = z_scores.replace([np.inf, -np.inf, np.nan], 0)
    z_scores_no_cutoff = z_scores_no_cutoff.replace([np.inf, -np.inf, np.nan], 0)

    return z_scores, z_scores_no_cutoff, feature_stats


def normalize_row(row):
    # Check for existing NaN or inf values
    if row.isnull().any() or np.isinf(row).any():
        # Replace with 0 for normalization
        row = row.replace([np.inf, -np.inf, np.nan], 0)
    
    # Min max scaling
    scale = max(abs(row.max()), abs(row.min()))
    
    # Handle division by zero
    if scale == 0:
        return row.fillna(0).astype(int)
    
    return ((row / scale) * 10).round().astype(int)


def format_json(df, bio_features):
    json_structure = {}

    for layer_head, series in df.iterrows():
        series_dict = {key: (int(value) if isinstance(value, np.integer) else value) for key, value in series.to_dict().items()}
        # Coefficients that are not significantly different from zero are removed
        non_zero_coeffs = {k: v for k, v in series_dict.items() if v != 0 and k in bio_features}
        sentences = [[feature if series_dict[feature] == 0 else [feature, series_dict[feature]] for feature in bio_features if 'position' not in feature]]
        
        json_structure[layer_head] = {
            "name": layer_head,
            "given_name": "...",
            "explanation": "The main thing this head does is find...",
            "sentences": sentences
        }

    json_output = json.dumps(json_structure, indent=4)
    return json_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/coef/",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/",
        type=str,
        help="The full path to the output directory",
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
    args = parser.parse_args()

    data_path = args.data_path
    full_path = args.full_path
    model_name = args.model_name
    model_subtype = args.model_subtype
    
    # Make sure if model_subtype selected as "random", the model_name is DNABERT_TATA or DNABERT_enhancer
    if args.model_subtype == "random":
        assert args.model_name == "DNABERT_TATA" or args.model_name == "DNABERT_enhancers", "Model name should be DNABERT_TATA or DNABERT_enhancers"
    
    # Path to the coefficient file
    coef_path = f'{data_path}{model_name}/{model_subtype}_results.csv'
    
    # Model to distribution folder mapping
    model_to_folder = {
        'DNABERT_TATA': 'DNABERT_TATA',
        'DNABERT_enhancers': 'DNABERT_enhancers',
        'scgpt_ms': 'scgpt_ms',
        'scgpt_pancreas': 'scgpt_pancreas',
        'NT_TATA': 'NT_TATA',
        'NT_enhancers': 'NT_enhancers'
    }
    
    # Get folder for distribution data
    folder = model_to_folder.get(model_name, model_name)
    distribution_folder = f'{full_path}/data/distributions/{folder}/'
    
    # Check if distribution folder exists
    if not os.path.exists(distribution_folder):
        print(f"Warning: Distribution folder {distribution_folder} does not exist")
        try:
            os.makedirs(distribution_folder, exist_ok=True)
            print(f"Created distribution folder {distribution_folder}")
        except Exception as e:
            print(f"Error creating distribution folder: {str(e)}")
            return
    
    # Read coefficient data
    try:
        real_coef_df = pd.read_csv(coef_path, index_col=0)
        # Exclude positional columns
        real_coef_df = real_coef_df[[col for col in real_coef_df.columns if 'position' not in col]]
    except Exception as e:
        print(f"Error reading coefficient file {coef_path}: {str(e)}")
        return

    # Calculate Z-scores with 3-sigma thresholds
    z_scores_3sigma, z_scores_no_cutoff, feature_stats = calculate_feature_specific_thresholds(
        real_coef_df, distribution_folder
    )
    
    # Create output directories
    os.makedirs(f'{full_path}/data/z_scores/', exist_ok=True)
    os.makedirs(f'{full_path}/data/z_scores/{model_name}', exist_ok=True)
    os.makedirs(f'{full_path}/data/coef/', exist_ok=True)
    os.makedirs(f'{full_path}/data/coef/{model_name}', exist_ok=True)
    os.makedirs(f'{full_path}/data/explanation_prompts/{model_name}', exist_ok=True)
    
    # Save Z-score versions
    z_scores_no_cutoff.to_csv(f'{full_path}/data/z_scores/{model_name}/{model_subtype}_z_scores_no_cutoff.csv')
    z_scores_3sigma.to_csv(f'{full_path}/data/z_scores/{model_name}/{model_subtype}_z_scores_3sigma.csv')
    
    # Save feature statistics
    feature_stats_df = pd.DataFrame(feature_stats).T
    feature_stats_df.to_csv(f'{full_path}/data/z_scores/{model_name}/{model_subtype}_feature_thresholds.csv')
    
    # # Normalize coefficients
    # normalized_3sigma = z_scores_3sigma.apply(normalize_row, axis=0)
    
    # # Save normalized results
    # normalized_3sigma.to_csv(f'{full_path}/data/coef/{model_name}/{model_subtype}_normalized_3sigma.csv')
    
    # Create JSON prompts for 3-sigma threshold
    bio_features = [col for col in real_coef_df.columns if 'position' not in col]
    
    # 3-sigma threshold JSON
    json_output_3sigma = format_json(z_scores_3sigma, bio_features)
    with open(f'{full_path}/data/explanation_prompts/{model_name}/{model_subtype}.json', 'w') as f:
        f.write(json_output_3sigma)
    
    print(f"3-sigma threshold analysis complete for {model_name} ({model_subtype})")

if __name__ == "__main__":
    main()