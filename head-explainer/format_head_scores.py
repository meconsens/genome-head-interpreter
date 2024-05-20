import pandas as pd
import csv
import numpy as np
import argparse
import json
import os


def normalize_row(row):
    # absolute maximum to use as the scaling factor
    scale = max(row.max(), abs(row.min()))
    return ((row / scale) * 10).round().astype(int)
def format_json(df):
    bio_features = [col for col in df.columns if not (('mse' in col or 'r2' in col) )]
    json_structure = {}

    for layer_head, series in df.iterrows():
        #now a dict
        series_dict = series.to_dict()
        
        # handle int64 
        series_dict = {key: (int(value) if isinstance(value, np.integer) else value) for key, value in series_dict.items()}

        # filter out zero coeff
        non_zero_coeffs = {k: v for k, v in series_dict.items() if v != 0}

        # "sentences"
        sentences = [[feature if series_dict[feature] == 0 else [feature, series_dict[feature]] for feature in bio_features]]
        
        # JSON for the current layer_head
        json_structure[layer_head] = {
            "name": layer_head,
            "explanation": "The main thing this head does is find...",
            "sentences": sentences
        }

    # dictionary to JSON string
    json_output = json.dumps(json_structure, indent=4)

    return json_output



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/coef/DNABERT_results_full.csv",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/head-explainer/",
        type=str,
        help="The full path to the format_head_scores.py file",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT",
        type=str,
        help="The model being explained",
    )
    args = parser.parse_args()

    data_path = args.data_path
    full_path = args.full_path
    model_name = args.model_name
    
    df_results = pd.read_csv(data_path, index_col=0)
    bio_features = [col for col in df_results.columns if not (('layer_head' in col) )]
    layer_heads = df_results['layer_head'].copy()
    normalized_rounded_df = df_results[bio_features].apply(normalize_row, axis=1)
    normalized_rounded_df.insert(0, 'layer_head', layer_heads)

    os.makedirs(f'{full_path}/data/coef/', exist_ok=True)
    normalized_rounded_df.to_csv(f'{full_path}/data/coef/{model_name}_norm_results_full.csv', index=False)
    
    os.makedirs(f'{full_path}/data/explanation_prompts/', exist_ok=True)
    json_format = format_json(normalized_rounded_df)
    with open(f'{full_path}/data/explanation_prompts/{model_name}.json', 'w') as f:
        f.write(json_format)



if __name__ == "__main__":
    main()