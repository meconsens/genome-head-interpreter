import pandas as pd
import numpy as np
import argparse
from scipy import stats
import os
from config_functions import configure_DNABERT, configure_scgpt

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
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/scores/DNABERT_kmer_scores.csv",
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
        default="DNABERT",
        type=str,
        help="The model being explained",
    )
    args = parser.parse_args()

    data_path = args.data_path
    full_path = args.full_path
    model = args.model_name
    
    df = pd.read_csv(data_path, sep=';')
    
    config_functions = {
        'DNABERT': configure_DNABERT,
        'DNABERT_pretrained': configure_DNABERT,
        'DNABERT_random': configure_DNABERT,
        'DNABERT_random_init': configure_DNABERT,
        'DNABERT_TATA': configure_DNABERT
    }

    config_function = config_functions.get(args.model_name, configure_scgpt)
    
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
    coef_df.to_csv(f'{full_path}/data/coef/coef_{model}_results.csv')

if __name__ == "__main__":
    main()