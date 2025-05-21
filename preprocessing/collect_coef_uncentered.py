import pandas as pd
import numpy as np
import argparse
from scipy import stats
import os
from config_functions import configure_DNABERT, configure_scgpt, configure_nucleotide_transformer

def run_global_uncentered_correlations(df, attention_score_columns, bio_feature_columns, seq_lengths):
    global_coef_dict = {}

    for layer_head in attention_score_columns:
        X = None
        Y = None

        for index, seq in df.iterrows():
            seq_len = seq_lengths[index] if isinstance(seq_lengths, list) else seq_lengths
            x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((len(bio_feature_columns), seq_len)).T
            y = seq[layer_head]

            X = np.concatenate([X, x]) if X is not None else x
            Y = np.concatenate([Y, y]) if Y is not None else y

        if len(X) != len(Y):
            print(f"WARNING: Global shape mismatch for {layer_head}: X={len(X)}, Y={len(Y)}")
            global_coef_dict[layer_head] = [0] * len(bio_feature_columns)
            continue

        X = np.nan_to_num(X)
        global_coef_dict[layer_head] = [
            stats.spearmanr(X[:, col], Y).statistic if not np.isnan(stats.spearmanr(X[:, col], Y).statistic) else 0
            for col in range(X.shape[1])
        ]

    return global_coef_dict

def run_label_uncentered_correlations(df, attention_score_columns, bio_feature_columns, seq_lengths, label_column='label'):
    label_coef_dict = {}
    df['simple_label'] = df[label_column].apply(lambda x: str(x).split(',')[0])
    unique_labels = df['simple_label'].unique()

    for layer_head in attention_score_columns:
        for label in unique_labels:
            subset_df = df[df['simple_label'] == label]
            if subset_df.empty:
                continue

            key = f"{layer_head}_{label}"
            label_coef_dict[key] = []

            X = None
            Y = None

            for index, seq in subset_df.iterrows():
                seq_len = seq_lengths[index] if isinstance(seq_lengths, list) else seq_lengths
                x = np.concatenate(seq[bio_feature_columns].values).ravel().reshape((len(bio_feature_columns), seq_len)).T
                y = seq[layer_head]

                X = np.concatenate([X, x]) if X is not None else x
                Y = np.concatenate([Y, y]) if Y is not None else y

            if len(X) != len(Y):
                print(f"WARNING: Shape mismatch for {key}: X={len(X)}, Y={len(Y)}")
                label_coef_dict[key] = [0] * len(bio_feature_columns)
                continue

            X = np.nan_to_num(X)
            label_coef_dict[key] = [
                stats.spearmanr(X[:, col], Y).statistic if not np.isnan(stats.spearmanr(X[:, col], Y).statistic) else 0
                for col in range(X.shape[1])
            ]

    return label_coef_dict

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

    data_path = f'{args.data_path}{args.model_name}/{args.model_name}_{args.model_subtype}_scores.csv'
    df = pd.read_csv(data_path, sep=';')

    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")

    minmax_cols = [col for col in df.columns if '_minmax' in col]
    if minmax_cols:
        df = df.drop(columns=minmax_cols)

    config_map = {
        'DNABERT_TATA': configure_DNABERT,
        'DNABERT_enhancers': configure_DNABERT,
        'scgpt_ms': configure_scgpt,
        'scgpt_pancreas': configure_scgpt,
        'NT_TATA': configure_nucleotide_transformer,
        'NT_enhancers': configure_nucleotide_transformer
    }

    config = config_map[args.model_name](df)
    df = config.get("transformed_df", df)
    attention_score_columns = config["attention_score_columns"]
    bio_feature_columns = config["bio_feature_columns"]
    seq_length = config["seq_length"]

    os.makedirs(f'{args.full_path}/data/coef/{args.model_name}-UNCENTERED', exist_ok=True)

    # Global correlations
    global_coef_dict = run_global_uncentered_correlations(df, attention_score_columns, bio_feature_columns, seq_length)
    global_df = pd.DataFrame.from_dict(global_coef_dict, orient="index", columns=bio_feature_columns)
    global_df.to_csv(f'{args.full_path}/data/coef/{args.model_name}-UNCENTERED/{args.model_subtype}_global_headcorr.csv')

    # Label-specific correlations
    label_coef_dict = run_label_uncentered_correlations(df, attention_score_columns, bio_feature_columns, seq_length, args.label_column)
    all_labels = df['simple_label'].unique()

    for label in all_labels:
        label_keys = [k for k in label_coef_dict if k.endswith(f"_{label}")]
        if not label_keys:
            continue
        heads = [k.rsplit('_', 1)[0] for k in label_keys]
        data = {head: label_coef_dict[f"{head}_{label}"] for head in heads}
        label_df = pd.DataFrame.from_dict(data, orient="index", columns=bio_feature_columns)
        label_df.to_csv(f'{args.full_path}/data/coef/{args.model_name}-UNCENTERED/{args.model_subtype}_label_{label}_headcorr.csv')

    print(f"Saved global and per-label uncentered correlation results to {args.full_path}/data/coef/{args.model_name}-UNCENTERED/")

if __name__ == "__main__":
    main()
