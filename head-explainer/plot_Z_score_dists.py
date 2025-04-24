import pandas as pd
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt

def calculate_z_scores(real_df, distribution_folder):
    #find the coef distribution folder
    files = [f for f in os.listdir(distribution_folder) if f.startswith('coef_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No coefficient files found in the distribution folder.")
    
    print(f"Found {len(files)} distribution files in {distribution_folder}")
    
    try:
        dfs = [pd.read_csv(os.path.join(distribution_folder, file), index_col=0) for file in files]
        dist_df = pd.concat(dfs, axis=1)
        dist_df = dist_df[[col for col in dist_df.columns if 'position' not in col]]
        
        z_scores = pd.DataFrame(index=real_df.index)  # dataFrame to store Z-scores
        for feature in real_df.columns:
            # Check if feature exists in distribution dataframe
            if feature in dist_df.columns:
                mean = dist_df[feature].mean(axis=1)
                std = dist_df[feature].std(axis=1)
                
                # Handle zero standard deviations
                std[std == 0] = np.nan  # 0 standard deviations with NaN to avoid division by zero
                
                # Calculate z-scores with protection against NaN and inf values
                z_scores[feature] = (real_df[feature] - mean) / std
                
                # Report statistics
                nan_count = z_scores[feature].isna().sum()
                inf_count = np.isinf(z_scores[feature]).sum()
                print(f"Feature {feature}: {nan_count} NaN values, {inf_count} infinite values")
            else:
                print(f"Warning: Feature '{feature}' not found in distribution data, skipping")
                z_scores[feature] = np.nan
    
    except Exception as e:
        print(f"Error during Z-score calculation: {str(e)}")
        # Return empty dataframe with same structure as real_df
        return pd.DataFrame(index=real_df.index, columns=real_df.columns)

    return z_scores

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
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/head-explainer/",
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
    
    #make sure if model_subtype selected as "random", the model_name is DNABERT_TATA or DNABERT_enhancer
    if args.model_subtype == "random":
        assert args.model_name == "DNABERT_TATA" or args.model_name == "DNABERT_enhancers", "Model name should be DNABERT_TATA or DNABERT_enhancers"
    
    # Path to the coefficient file
    coef_path = f'{data_path}{model_name}/{model_subtype}_results.csv'
    
    # Model to distribution folder mapping
    model_to_folder = {
        'DNABERT_TATA': 'DNABERT_TATA',
        'DNABERT_fake_TATA': 'DNABERT_fake_TATA',
        'DNABERT_enhancers': 'DNABERT_enhancers',
        'scgpt_ms': 'scgpt_ms',
        'scgpt_pancreas': 'scgpt_pancreas',
        'NT_fake_TATA': 'NT_fake_TATA',
        'NT_TATA': 'NT_TATA',
        'NT_enhancers': 'NT_enhancers'
    }
    
    # Get folder for distribution data
    folder = model_to_folder.get(model_name, model_name)
    distribution_folder = f'{full_path}/preprocessing/data/distributions/{folder}/'
    
    # Read coefficient data
    real_coef_df = pd.read_csv(coef_path, index_col=0)
    # exclude positional columns
    real_coef_df = real_coef_df[[col for col in real_coef_df.columns if 'position' not in col]]

    # Calculate Z-scores
    z_scores = calculate_z_scores(real_coef_df, distribution_folder)

    # Create output directory
    os.makedirs(f'{full_path}/preprocessing/data/plots/{model_name}/', exist_ok=True)
    
    # Only plot if model_subtype is "finetuned"
    if model_subtype == "finetuned":
        # Plot all features on the same plot
        plt.figure(figsize=(12, 8))
        for feature in z_scores.columns:
            plt.hist(z_scores[feature], alpha=0.5, bins=20, label=feature)
        
        plt.title(f'Z-scores Distribution for {model_name} ({model_subtype})')
        plt.xlabel('Z-score')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{full_path}/preprocessing/data/plots/{model_name}/{model_subtype}_all_features_z_scores.png')
        plt.close()
    
    # Save Z-scores to file
    z_scores.to_csv(f'{full_path}/preprocessing/data/plots/{model_name}/{model_subtype}_z_scores.csv')

if __name__ == "__main__":
    main()