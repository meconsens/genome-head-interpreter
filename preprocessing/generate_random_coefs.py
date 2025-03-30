import pandas as pd
import csv
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import time
from config_functions import configure_DNABERT, configure_scgpt, configure_nucleotide_transformer

def shuffle_attention_scores(df, attention_score_columns, block_size=10):
    # shuffle attentions between sequences
    for col in attention_score_columns:
        shuffled_data = df[col].sample(frac=1).reset_index(drop=True)
        df[col] = shuffled_data

    # now make blocks of 10 (roughly) and shuffle WITHIN sequence
    if attention_score_columns:  
        for col in attention_score_columns:
            #shuffle each seq
            for index, row in df.iterrows():
                sequence = np.array(row[col])
                seq_len = len(sequence)
                #print(f'seq len: {seq_len}')
                full_blocks = seq_len // block_size
                #print(f'full blocks: {full_blocks}')
                num_elements = full_blocks * block_size
                #print(f'num elements: {num_elements}')

                # split the sequence into blocks and remainder
                full_sequence = sequence[:num_elements]
                remainder = sequence[num_elements:]

                # reshape full blocks and shuffle
                if full_blocks > 0:
                    blocks = full_sequence.reshape(full_blocks, block_size)
                    np.random.shuffle(blocks)  
                    full_sequence = blocks.flatten()
                
                # shuffle remainder
                np.random.shuffle(remainder)

                # concatenate shuffle blocks with shuffle remainder
                df.at[index, col] = np.concatenate((remainder, full_sequence))

    return df

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


def parallel_shuffle_and_analysis(df, attention_score_columns, bio_feature_columns, seq_length, model, full_path, iteration):
    # seed for each process 
    pid = os.getpid()
    current_time = time.time()
    unique_seed = int((pid + current_time) * 1000) % 4294967295 #make sure different processes have unique seed...

    np.random.seed(unique_seed)

    # shuffling
    shuffled_df = shuffle_attention_scores(df, attention_score_columns)
    coef_dict = run_spearman(shuffled_df, attention_score_columns, bio_feature_columns, seq_length)

    # save results
    os.makedirs(f'{full_path}/data/distributions/{model}/', exist_ok=True)
    results_filename = f'{full_path}/data/distributions/{model}/coef_{iteration}_results.csv'
    pd.DataFrame.from_dict(coef_dict, orient="index", columns=bio_feature_columns).to_csv(results_filename)
    print(f"Process {pid} with seed {unique_seed} completed and saved to {results_filename}")

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

    print(f"Attention score columns: {attention_score_columns}, Bio feature columns: {bio_feature_columns}, Sequence length: {seq_length}")

    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(parallel_shuffle_and_analysis, new_df, attention_score_columns, bio_feature_columns, seq_length, args.model_name, args.full_path, i) for i in range(100)]
        for future in as_completed(futures):
            future.result()

if __name__ == "__main__":
    main()