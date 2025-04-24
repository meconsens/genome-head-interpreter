import pandas as pd
import numpy as np
import argparse
from scipy import stats
import os
from config_functions import configure_DNABERT, configure_scgpt, configure_nucleotide_transformer

def run_spearman(df, attention_score_columns, bio_feature_columns, seq_lengths):
    coef_dict = {}
    for layer_head in attention_score_columns:
        coef_dict[layer_head] = []

        X = None
        Y = None

        # loop over rows (sequences)
        for index, seq in df.iterrows():
            # seq_lengths is a list (variable lengths) or an integer (fixed length)
            if isinstance(seq_lengths, list):
                seq_len = seq_lengths[index]
            else:
                seq_len = seq_lengths  # same length for all sequences

            # reshape according to the sequence length determined
            x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((len(bio_feature_columns), seq_len)).T
            y = seq[layer_head]

            if X is not None:
                X = np.concatenate([X, x])
                Y = np.concatenate([Y, y])
            else:
                X = x
                Y = y

        # Check if X and Y have matching first dimensions
        if len(X) != len(Y):
            print(f"WARNING: Shape mismatch in column '{layer_head}': X has {len(X)} rows, Y has {len(Y)} elements")
            coef_dict[layer_head] = [0] * len(bio_feature_columns)
            continue

        # handle NaN values
        X = np.nan_to_num(X)

        # Spearman correlation for each feature column
        for col_iter in range(X.shape[1]):
            result = stats.spearmanr(X[:, col_iter], Y)
            if not np.isnan(result.statistic):
                coef_dict[layer_head].append(result.statistic)
            else:
                coef_dict[layer_head].append(0)  # handle NaN

    return coef_dict


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
    args = parser.parse_args()

    data_path = args.data_path
    full_path = args.full_path
    model = args.model_name
    
    #make sure if model_subtype selected as "random", the model_name is DNABERT_TATA or DNABERT_enhancer
    if args.model_subtype == "random":
        assert args.model_name == "DNABERT_TATA" or args.model_name == "DNABERT_enhancers", "Model name should be DNABERT_TATA or DNABERT_enhancers"
    

    #add model name to the data path
    data_path = f'{data_path}{args.model_name}/{args.model_name}_{args.model_subtype}_scores.csv'
    
    df = pd.read_csv(data_path, sep=';')
    

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
    print(f'Bio Feature Columns: {bio_feature_columns}')
    print(f'Sequence Length: {seq_length}')

    coef_dict = run_spearman(new_df, attention_score_columns, bio_feature_columns, seq_length)

    coef_df = pd.DataFrame.from_dict(coef_dict, orient="index", columns=bio_feature_columns)

    # save
    os.makedirs(f'{full_path}/data/coef/', exist_ok=True)
    #make specific folder for each model
    os.makedirs(f'{full_path}/data/coef/{args.model_name}', exist_ok=True)
    coef_df.to_csv(f'{full_path}/data/coef/{args.model_name}/{args.model_subtype}_results.csv')

if __name__ == "__main__":
    main()