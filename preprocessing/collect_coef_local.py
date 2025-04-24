import pandas as pd
import numpy as np
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from scipy import stats
from config_functions import configure_DNABERT, configure_scgpt, configure_nucleotide_transformer
import json


def extract_and_calculate_correlations_for_feature(df, feature, seq_len, attention_score_columns, output_folder, model_name, model_subtype, identifier, seq_id_csv):
    print(f"\nExtracting attention score subsequences and correlations for feature: {feature}")
    window_size = 50  # window size

    if feature not in df.columns:
        print(f"Warning: Feature '{feature}' not found in DataFrame")
        return

    # dictionary to accumulate data for correlations
    accumulated_data = {layer_head: {'feature_values': [], 'attention_scores': []} for layer_head in attention_score_columns}

    saved_example_count = 0  # examples per feature
    seq_info_list = []  # store seq_id and actual feature value information
    if "DNABERT" in model_name or "NT" in model_name:
        seq_len_list = seq_len 
    # per sequence
    for index, row in df.iterrows():
        #if model name contains DNABERT, get the seq_len from the list
        if "DNABERT" in model_name or "NT" in model_name:
            seq_len = seq_len_list[index]
        if saved_example_count >= 200:
            print(f"Reached 200 examples for feature {feature}. Stopping further processing for this feature.")
            break

        feature_values = np.array(row[feature])
        seq_id = row[identifier]
        print(f"Feature values: {feature_values}")

        # positions where the feature is present (non-zero values)
        non_zero_positions = np.where(feature_values != 0)[0]
        print(len(non_zero_positions))

        if len(non_zero_positions) == 0:
            # whole sequence is zero (no feature present)
            print(f"No feature present in the sequence for index {index}. Skipping.")
            continue  # skip bc no feature is present

        elif len(non_zero_positions) == len(feature_values):
            # whole sequence has no zero values (feature always present to some degree)
            # find  most variable region
            var_window = 10
            variances = [np.var(feature_values[i:i + var_window]) for i in range(len(feature_values) - var_window + 1)]

            # window of max variance in feature values
            max_var_index = np.argmax(variances)
            #centre max variance, don't go out of bounds
            center_position = max(0, max_var_index + var_window // 2)

            # Fixing missing variable definition
            subsequence_start = max(0, center_position - window_size // 2)
            subsequence_end = min(seq_len, center_position + window_size // 2)
            
            print(f"Most variable region: {feature_values[subsequence_start:subsequence_end]}")

        else:
            # case with zero and non-zero (feature presence = 1)
            # find all contiguous regions of non-zero values
            non_zero_regions = np.split(non_zero_positions, np.where(np.diff(non_zero_positions) != 1)[0] + 1)

            # get longest contiguous region of non-zero values
            longest_region = max(non_zero_regions, key=len)
            start = longest_region[0]
            end = longest_region[-1] + 1  # end is one past the last non-zero value
            center_position = (start + end) // 2

        # define window bounds, extending evenly on each side if possible
        flank_size = window_size // 2
        subsequence_start = max(0, center_position - flank_size)
        subsequence_end = min(seq_len, center_position + flank_size)

        # adjust if near sequence bounds
        if subsequence_end - subsequence_start < window_size:
            if subsequence_start == 0:
                subsequence_end = min(seq_len, window_size)
            elif subsequence_end == seq_len:
                subsequence_start = max(0, seq_len - window_size)

        # Handle the case where 'start' and 'end' might be undefined in some code paths
        if 'start' not in locals() or 'end' not in locals():
            start = subsequence_start
            end = subsequence_end

        feature_length = end - start
        left_flank = max(0, start - subsequence_start)
        right_flank = max(0, subsequence_end - end)

        print(f"Sequence {index}: {subsequence_start} - {subsequence_end}")
        print(f"Feature Length: {feature_length}, Left Flank: {left_flank}, Right Flank: {right_flank}")

        # get feature values for the subsequence
        seq_feature_values = feature_values[subsequence_start:subsequence_end]
        print("seq_feature_values: ", seq_feature_values)
        
        seq_info_list.append({
            'seq_id': seq_id,
            'seq_feature_values': json.dumps(seq_feature_values.tolist()),  #JSON string
            'feature': feature
        })

         # accumulate data for correlations
        for layer_head in attention_score_columns:
            attention_subseq = row[layer_head][subsequence_start:subsequence_end]
            accumulated_data[layer_head]['feature_values'].extend(feature_values[subsequence_start:subsequence_end])
            accumulated_data[layer_head]['attention_scores'].extend(attention_subseq)

        #example counter +
        saved_example_count += 1
    
    if saved_example_count < 50:
        print(f"Skipping feature {feature} as it has only {saved_example_count} examples (minimum required: 50).")
        return
    
    # dictionary to store the final correlation coefficients
    correlation_dict = {'feature': [feature]}

    # calculate the correlation across all subsequences for each layer-head
    for layer_head, data in accumulated_data.items():
        if data['feature_values']:  # data to correlate
            result = stats.spearmanr(data['feature_values'], data['attention_scores'])
            correlation = result.correlation if not np.isnan(result.correlation) else 0
            correlation_dict[layer_head] = [correlation]
        else:
            correlation_dict[layer_head] = [np.nan]

    # df
    feature_coef_df = pd.DataFrame(correlation_dict)

    # save to csv
    feature_name = feature.replace("/", "+")
    #ensure os.mkdir exist
    os.makedirs(f'{output_folder}/{model_name}', exist_ok=True)
    output_csv = os.path.join(output_folder, f'{model_name}/{model_subtype}_{feature_name}_correlations.csv')
    feature_coef_df.to_csv(output_csv, index=False)
    print(f"Saved correlation coefficients for feature {feature} to {output_csv}")
    print(f"Saved example count:{saved_example_count}")

    # save seq_id and seq_feature_values information
    seq_info_df = pd.DataFrame(seq_info_list)
    seq_info_df.to_csv(seq_id_csv, mode='a', header=not os.path.exists(seq_id_csv), index=False)
    print(f"Saved seq_id and seq_feature_values information to {seq_id_csv}")
    return output_csv


def process_feature_in_parallel(config, feature, output_folder, model_name, model_subtype, identifier, seq_id_csv):
    return extract_and_calculate_correlations_for_feature(config['transformed_df'], feature, config['seq_length'], config['attention_score_columns'], output_folder, model_name, model_subtype, identifier, seq_id_csv)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="",
        type=str,
        help="The path to the data (optional, constructed from model_name and subtype if not provided)",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/",
        type=str,
        help="The full path to the script directory",
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

    full_path = args.full_path
    model_name = args.model_name
    model_subtype = args.model_subtype

    #make sure if model_subtype selected as "random", the model_name is DNABERT_TATA or DNABERT_enhancer
    if args.model_subtype == "random":
        assert args.model_name == "DNABERT_TATA" or args.model_name == "DNABERT_enhancers", "Model name should be DNABERT_TATA or DNABERT_enhancers"
    
    
    # Build data_path if not provided
    if not args.data_path:
        data_path = f'{full_path}/data/scores/{model_name}/{model_name}_{model_subtype}_scores.csv'
    else:
        data_path = args.data_path
    
    print(f"Using data path: {data_path}")
    
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

    # configs
    config_functions = {
        'DNABERT_TATA': configure_DNABERT,
        'DNABERT_enhancers': configure_DNABERT,
        'scgpt_ms': configure_scgpt,
        'scgpt_pancreas': configure_scgpt,
        'NT_TATA': configure_nucleotide_transformer,
        'NT_enhancers': configure_nucleotide_transformer
    }

    # Get the appropriate config function
    if model_name in config_functions:
        config_function = config_functions[model_name]
    else:
        print(f"Warning: No specific configuration for model {model_name}. Using scGPT configuration.")
        config_function = configure_scgpt
    
    # configuration for the run
    config = config_function(df)

    new_df = config["transformed_df"] if "transformed_df" in config else df

    def check_types(df):
        for col in df.columns:
            col_types = df[col].apply(lambda x: type(x).__name__)
            print(f"Column '{col}' contains types: {col_types.unique()}")

    check_types(new_df)
    
    attention_score_columns = config["attention_score_columns"]
    bio_feature_columns = config["bio_feature_columns"]
    seq_length = config["seq_length"]
    seq_column = config['seq_column']

    # do not include biological features containing the word 'position'
    bio_feature_columns = [col for col in bio_feature_columns if 'position' not in col]

    output_folder = f'{full_path}/data/coef/local/'
    #makedirs outputfolder
    os.makedirs(output_folder, exist_ok=True)
    seq_id_csv = os.path.join(output_folder, f'{model_name}/{model_name}_{model_subtype}_seq_info.csv')

    #ensure os.mkdir exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f'{output_folder}/{model_name}', exist_ok=True)

    # parallelize
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(
            process_feature_in_parallel, 
            config, 
            feature, 
            output_folder, 
            model_name, 
            model_subtype, 
            seq_column, 
            seq_id_csv
        ) for feature in bio_feature_columns]
        
        # wait for all features to be processed and collect the CSV paths
        csv_files = [future.result() for future in futures if future.result() is not None]

    if not csv_files:
        print("No CSV files were generated. Check if any features were processed successfully.")
        return

    # merge all individual feature CSVs
    merged_df = pd.concat([pd.read_csv(csv) for csv in csv_files], ignore_index=True)

    # save
    final_output_csv = os.path.join(output_folder, f'{model_name}/{model_subtype}_coef.csv')
    merged_df.to_csv(final_output_csv, index=False)
    print(f"Saved combined correlation coefficients to {final_output_csv}")

if __name__ == "__main__":
    main()