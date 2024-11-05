import pandas as pd
import numpy as np
import random
from scipy import stats
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from config_functions import configure_DNABERT, configure_enformer, configure_scgpt
import json

def get_subsequence_vals_from_csv(seq_info_csv):
    subsequence_vals = {}
    df = pd.read_csv(seq_info_csv)

    for _, row in df.iterrows():
        feature = row['feature']
        seq_id = row['seq_id']
        seq_feature_values = row['seq_feature_values']

        if feature not in subsequence_vals:
            subsequence_vals[feature] = {}

        # json string to list
        seq_feature_values_list = json.loads(seq_feature_values)
        subsequence_vals[feature][seq_id] = seq_feature_values_list  # Store the actual feature values

    return subsequence_vals


def sample_random_attention_scores(df, full_sequence_length, subsequence_length, seq_column, attention_heads):
    sampled_attention_scores = []
    #get other sequences of greater or equal sequence length to full_sequence_length
    filtered_df = df[df[attention_heads[0]].apply(lambda x: len(x) >= full_sequence_length)]
    if not filtered_df.empty:
        # select a random row from the filtered df
        random_row = filtered_df.sample(n=1).iloc[0]
        
        # get full sequence length of the selected random sequence
        seq_len = len(random_row[attention_heads[0]])
        
        if seq_len >= subsequence_length:
            start = random.randint(0, seq_len - subsequence_length)
            
            random_attention_scores = {}
            for layer_head in attention_heads:
                random_attention_scores[layer_head] = random_row[layer_head][start:start + subsequence_length]
            sampled_attention_scores.append(random_attention_scores)
    
    return sampled_attention_scores


# spearman correlation for given feature using real feature values and random attention scores
def calculate_correlations(feature_name, feature_values_list, random_attention_scores, attention_heads):
    # combine feature values
    feature_values_combined = np.concatenate(feature_values_list)
    print(f"feature_values_combined: {len(feature_values_combined)}")
    
    if not random_attention_scores:
        print("random_attention_scores is empty")
    
    print(f"random_attention_scores: {len(random_attention_scores)}")
    
    # dictionary to store correlations
    correlation_dict = {'feature': feature_name}
    
    # calculate correlations for each attention layer and head
    for head in attention_heads:
        attention_values_combined = np.concatenate([score[head] for score in random_attention_scores if head in score])
        print(f"attention_values_combined for {head}: {len(attention_values_combined)}")
        
        result = stats.spearmanr(feature_values_combined, attention_values_combined)
        correlation = result.correlation if not np.isnan(result.correlation) else 0
        correlation_dict[head] = correlation

    correlation_df = pd.DataFrame([correlation_dict])
    return correlation_df


#each iteration function
def random_sampling_iteration(df, attention_heads, subsequence_vals, iteration, output_folder, seq_column, model_name):
    all_correlations = []
    print('in random_sampling_iteration')
    df.set_index(seq_column, inplace=True, drop=False)
    
    # for all biological features in the CSV file
    for feature_name, seq_vals in subsequence_vals.items():
        print(f'Processing feature {feature_name} in iteration {iteration}')
        random_attention_scores = []
        feature_values_list = []
        
        # load the feature values from the CSV info and sample random attention scores
        for seq_id, feature_values in seq_vals.items():
            feature_values = np.array(feature_values)  # np array
            feature_values_list.append(feature_values)

            #print(f'feature_values: {len(feature_values)}')
                
            # sample random attention scores of the same length from a random portion of the CSV data
            subseq_len = len(feature_values)
            print("SUBSEQ LEN:", subseq_len)
            print("MODEL_NAME:", model_name)
            if model_name in ['scgpt_ms', 'scgpt_pancreas']:
                full_sequence_length = df.loc[seq_id]['layer0_head0'].shape[0]
            else:
                full_sequence_length = df.loc[seq_id]['layer0-head0'].shape[0]
            random_attention = sample_random_attention_scores(df, full_sequence_length, subseq_len, seq_column, attention_heads)
            random_attention_scores.extend(random_attention)
        
        # spearman correlations for this feature
        feature_correlation_df = calculate_correlations(feature_name, feature_values_list, random_attention_scores, attention_heads)
        all_correlations.append(feature_correlation_df)
    
    print("about to merge all correlations for iteration")
    # merge all feature correlations for this iteration
    iteration_df = pd.concat(all_correlations, ignore_index=True)

    # save the correlation results for this iteration to a CSV
    iteration_output_csv = os.path.join(output_folder, f'{iteration}_coef.csv')
    iteration_df.to_csv(iteration_output_csv, index=False)
    print(f"Iteration {iteration} correlations saved to {iteration_output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/coef/DNABERT/DNABERT_seq_info.csv",
        type=str,
        help="The path to the seq info data file",
    )
    parser.add_argument(
        "--scores_csv",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/scores/DNABERT_scores.csv",
        type=str,
        help="The path to the CSV file containing the sequences and scores",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/",
        type=str,
        help="The full path for saving results",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT",
        type=str,
        help="The model being explained",
    )
    parser.add_argument(
        "--num_iterations",
        default=100,
        type=int,
        help="The number of random sampling iterations to run",
    )
    args = parser.parse_args()

    model = args.model_name
    seq_info_csv = args.data_path
    scores_csv_path = args.scores_csv
    output_folder = f"{args.full_path}/data/distributions/{model}/"
    
   
    os.makedirs(output_folder, exist_ok=True)

   
    subsequence_vals = get_subsequence_vals_from_csv(seq_info_csv)

    
    df = pd.read_csv(scores_csv_path, sep=';')
    #configs
    config_functions = {
        'DNABERT': configure_DNABERT,
        'DNABERT_pretrained': configure_DNABERT,
        'DNABERT_random': configure_DNABERT,
        'DNABERT_random_init': configure_DNABERT,
        'DNABERT_TATA': configure_DNABERT,
        'enformer': configure_enformer,
        'enformer_random_init': configure_enformer,
    }
    config_function = config_functions.get(model, configure_scgpt)
    
    config = config_function(df)

    new_df = config["transformed_df"] if "transformed_df" in config else df

    attention_heads = config["attention_score_columns"]
    seq_column = config['seq_column']

    attention_heads = [col for col in df.columns if 'layer' in col and 'head' in col]

    
    num_iterations = args.num_iterations
    #each iteration in parallel
    with ProcessPoolExecutor(max_workers=50) as executor:
        futures = [
            executor.submit(random_sampling_iteration, new_df, attention_heads, subsequence_vals, iteration, output_folder, seq_column, model)
            for iteration in range(num_iterations)
        ]

        for future in as_completed(futures):
            future.result()  # wait for the result

if __name__ == "__main__":
    main()


