import pandas as pd
import numpy as np
import argparse
import os
import json
import numpy as np
import pandas as pd

def calculate_z_scores(real_df, distribution_folder, task_name, model_name):
    #find the coef distribution folder
    if 'local' in task_name:
        files = [f for f in os.listdir(distribution_folder) if f.endswith('_coef.csv')]
    else:
        files = [f for f in os.listdir(distribution_folder) if f.startswith('coef_') and f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No coefficient files found in the distribution folder.")
    
    dfs = [pd.read_csv(os.path.join(distribution_folder, file), index_col=0) for file in files]
    dist_df = pd.concat(dfs, axis=1)
    dist_df = dist_df[[col for col in dist_df.columns if 'position' not in col]]
    
    # dictionary to store Z-scores
    z_scores_dict = {}

    #save another copy of the z scores without cut off
    z_scores_dict_no_cutoff = {}
    
    # Z-scores for each feature
    for feature in real_df.columns:
        mean = dist_df[feature].mean(axis=1)
        std = dist_df[feature].std(axis=1)
        std[std == 0] = np.nan  
        z_scores_dict[feature] = (real_df[feature] - mean) / std  
        z_scores_dict_no_cutoff[feature] = (real_df[feature] - mean) / std
        cut_off = 4.5
        z_scores_dict[feature].loc[z_scores_dict[feature].abs() <= cut_off] = 0
    
    z_scores = pd.DataFrame(z_scores_dict, index=real_df.index)
    z_scores_no_cutoff = pd.DataFrame(z_scores_dict_no_cutoff, index=real_df.index)
    
    # replace NaN and infinite values with 0
    z_scores = z_scores.replace([np.inf, -np.inf, np.nan], 0)
    z_scores_no_cutoff = z_scores_no_cutoff.replace([np.inf, -np.inf, np.nan], 0)

    return z_scores, z_scores_no_cutoff


def normalize_row(row):
    # check for existing NaN or inf values
    if row.isnull().any() or np.isinf(row).any():
        raise ValueError("Input row contains NaN or inf values.")
    
    # min max scaling
    scale = max(row.max(), abs(row.min()))
    print(f"scale: {scale}")
    
    # handle division by zero
    if scale == 0:
        return row.fillna(0).astype(int)
    
    return ((row / scale) * 10).round().astype(int)


def format_json(df, task_name):
    #flip the dataframe 
    print("task_name pre flip:", task_name) 
    #if task_name is local
    if 'local' in task_name:
        df = df.T
    bio_features = [col for col in df.columns if 'layer' not in col]
    print("BIO FEATURES:", bio_features)
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
    parser.add_argument("--full_path", default = "/scratch/ssd004/scratch/mconsens/genome-head-interpreter/head-explainer/", type=str, help="The full path to the output directory")
    parser.add_argument("--model_name", default="DNABERT", type=str, help="The model being explained")
    parser.add_argument("--task_name", default="task", type=str, help="The task being explained")
    args = parser.parse_args()

    real_coef_path = args.coef_path
    full_path = args.full_path
    model_name = args.model_name
    task_name = args.task_name
    
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
    if 'global' in task_name:
        real_coef_df = real_coef_df[[col for col in real_coef_df.columns if 'position' not in col]]
    
    print(f"THE TASK NAME IS: {task_name}")

    z_scores, z_scores_no_cutoff = calculate_z_scores(real_coef_df, distribution_folder, task_name, model_name)
    
    #let's save the z_scores
    os.makedirs(f'{full_path}/data/z_scores/', exist_ok=True)
    if (task_name =='global'):
        z_scores_no_cutoff.to_csv(f'{full_path}/data/z_scores/{model_name}_z_scores.csv')
    else:
        os.makedirs(f'{full_path}/data/z_scores/{model_name}', exist_ok=True)
        z_scores_no_cutoff.to_csv(f'{full_path}/data/z_scores/{model_name}/{model_name}_z_scores.csv')
    
    normalized_coefs = z_scores.apply(normalize_row, axis=0)

    os.makedirs(f'{full_path}/data/coef/', exist_ok=True)
    if (task_name =='global'):
        normalized_coefs.to_csv(f'{full_path}/data/coef/{model_name}_normalized_results.csv')
    else:
        normalized_coefs.to_csv(f'{full_path}/data/coef/{model_name}/{model_name}_normalized_results.csv')

    os.makedirs(f'{full_path}/data/explanation_prompts/{task_name}', exist_ok=True)
    json_output = format_json(normalized_coefs, task_name)
    print(f'{full_path}/data/explanation_prompts/{model_name}.json')
    with open(f'{full_path}/data/explanation_prompts/{task_name}/{model_name}.json', 'w') as f:
        f.write(json_output)

if __name__ == "__main__":
    main()
