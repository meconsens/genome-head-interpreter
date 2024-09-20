import pandas as pd
import numpy as np
import argparse
import os
import json

def calculate_z_scores(real_df, distribution_folder):
    #find the coef distribution folder
    files = [f for f in os.listdir(distribution_folder) if f.startswith('coef_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No coefficient files found in the distribution folder.")
    
    dfs = [pd.read_csv(os.path.join(distribution_folder, file), index_col=0) for file in files]
    dist_df = pd.concat(dfs, axis=1)
    dist_df = dist_df[[col for col in dist_df.columns if 'position' not in col]]
    
    z_scores = pd.DataFrame(index=real_df.index)  # dataFrame to store Z-scores
    for feature in real_df.columns:
        mean = dist_df[feature].mean(axis=1)
        std = dist_df[feature].std(axis=1)
        std[std == 0] = np.nan  # 0 standard deviations with NaN to avoid division by zero
        z_scores[feature] = (real_df[feature] - mean) / std  # Z-score per feature
        # nonsignificant coefficients to 0 before normalization
        z_scores.loc[z_scores[feature].abs() <= 4, feature] = 0

    return z_scores


def normalize_row(row):
    # Check for existing NaN or inf values
    if row.isnull().any() or np.isinf(row).any():
        raise ValueError("Input row contains NaN or inf values.")
    
    # Absolute maximum to use as the scaling factor
    scale = max(row.max(), abs(row.min()))
    print(f"scale: {scale}")
    
    # handle division by zero
    if scale == 0:
        return row.fillna(0).astype(int)
    
    return ((row / scale) * 10).round().astype(int)


def format_json(df):
    bio_features = [col for col in df.columns if col not in ('layer_head') and 'position' not in col]
    json_structure = {}

    for layer_head, series in df.iterrows():
        series_dict = {key: (int(value) if isinstance(value, np.integer) else value) for key, value in series.to_dict().items()}
        # coefficients that are not significantly different from zero or abs(coef) <= 4 removed
        non_zero_coeffs = {k: v for k, v in series_dict.items() if abs(v) > 4 and k in bio_features}
        #do not include position...
        sentences = [[feature if abs(series_dict[feature]) <= 4 else [feature, series_dict[feature]] for feature in bio_features if 'position' not in feature]]
        
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
    parser.add_argument("--coef_path", type=str, help="Path to the coefficients CSV")
    parser.add_argument("--pval_path", type=str, help="Path to the p-values CSV")
    parser.add_argument("--adj_pval_path", type=str, help="Path to the adjusted p-values CSV")
    parser.add_argument("--full_path", default = "/scratch/ssd004/scratch/mconsens/genome-head-interpreter/head-explainer/", type=str, help="The full path to the output directory")
    parser.add_argument("--model_name", default="DNABERT", type=str, help="The model being explained")
    args = parser.parse_args()

    real_coef_path = args.coef_path
    full_path = args.full_path
    model_name = args.model_name
    
    model_name_to_folder = {
        'DNABERT': 'DNABERT',
        'DNABERT_pretrained': 'DNABERT',
        'DNABERT_random': 'DNABERT',
        'DNABERT_random_init': 'DNABERT',
        'enformer': 'enformer',
        'enformer_random_init': 'enformer',
        'scgpt_ms': 'scgpt_ms',
        'scgpt_ms_random_init': 'scgpt_ms',
        'scgpt_ms_pretrained': 'scgpt_ms',
        'scgpt_pancreas': 'scgpt_pancreas',
        'scgpt_pancreas_random_init': 'scgpt_pancreas',
        'scgpt_pancreas_pretrained': 'scgpt_pancreas',
    }

    model_name = args.model_name
    print("THE MODEL NAME IS:", model_name)
    folder = model_name_to_folder.get(model_name, 'scgpt_pancreas')
    print("THE FOLDER NAME IS:", folder)
    distribution_folder = f'{full_path}/data/distributions/{folder}/'

    real_coef_df = pd.read_csv(real_coef_path, index_col=0)
    # exclude positional columns from real_coef_df
    real_coef_df = real_coef_df[[col for col in real_coef_df.columns if 'position' not in col]]


    z_scores = calculate_z_scores(real_coef_df, distribution_folder)

    normalized_coefs = z_scores.apply(normalize_row, axis=0)

    os.makedirs(f'{full_path}/data/coef/', exist_ok=True)
    normalized_coefs.to_csv(f'{full_path}/data/coef/{model_name}_normalized_results.csv')

    os.makedirs(f'{full_path}/data/explanation_prompts/', exist_ok=True)
    json_output = format_json(normalized_coefs)
    print(f'{full_path}/data/explanation_prompts/{model_name}.json')
    with open(f'{full_path}/data/explanation_prompts/{model_name}.json', 'w') as f:
        f.write(json_output)

if __name__ == "__main__":
    main()
