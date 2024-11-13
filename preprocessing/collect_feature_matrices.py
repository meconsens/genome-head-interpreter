import pandas as pd
import numpy as np
import argparse
import os
import h5py
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock


def map_sequence_ids_to_identifiers(hdf5_file_path):
    seq_id_mapping = {}
    with h5py.File(hdf5_file_path, 'r') as hdf5_store:
        for feature_name in h5py.File(hdf5_file_path, 'r').keys():  # Bio feature groups
            for seq_id in hdf5_store[feature_name].keys():  # Iterate through sequence IDs
                # Fetch the sequence identifier (assuming it's stored in the 'seq_column')
                seq_identifier = hdf5_store[feature_name][seq_id]['seq_column'][()]
                seq_id_mapping[seq_id] = seq_identifier
    return seq_id_mapping


def check_existing_files(output_folder, model, bio_feature_columns):
    # HDF5 file path
    hdf5_file_path = f'{output_folder}/data/feature_scores/{model}/feature_scores.h5'
    
    # Check if the HDF5 file exists
    if not os.path.exists(hdf5_file_path):
        return bio_feature_columns  # If file doesn't exist, all features are missing
    
    missing_features = []
    
    # Open the HDF5 file to check for missing features
    with h5py.File(hdf5_file_path, 'r') as hdf5_store:
        for feature in bio_feature_columns:
            feature = feature.replace("/", "+")
            if feature not in hdf5_store:
                missing_features.append(feature)
    return missing_features  # Return list of missing features (if any)

# def extract_and_save_subsequences(df, bio_feature_columns, seq_len, attention_score_columns, seq_column, output_folder, model_name):
#     # folder for saving feature-specific matrices
#     path_addition = f'{model_name}/'
#     model_folder = os.path.join(output_folder, path_addition)
#     os.makedirs(model_folder, exist_ok=True)
    
#     hdf5_file = os.path.join(model_folder, f'feature_scores.h5')  # HDF5 file
#     with h5py.File(hdf5_file, 'w') as hdf5_store:
#         # for bio feature columns (each representing a feature)
#         for feature in bio_feature_columns:
#             feature_path = feature.replace("/", "+")
#             print(f"\nExtracting attention score subsequences for feature: {feature}")
#             if feature not in df.columns:
#                 print(f"Warning: Feature '{feature}' not found in DataFrame")
#                 continue

#             # group for the feature in the HDF5 file
#             feature_group = hdf5_store.create_group(feature_path)

#             saved_example_count = 0  # Initialize counter for saved examples per feature

#             # per sequence
#             for index, row in df.iterrows():

#                 if saved_example_count >= 200:
#                     print(f"Reached 200 examples for feature {feature}. Stopping further processing for this feature.")
#                     break

#                 feature_values = np.array(row[feature])
#                 print(f"Feature values: {feature_values}")

#                 # positions where the feature is present (non-zero values)
#                 non_zero_positions = np.where(feature_values != 0)[0]
#                 print(len(non_zero_positions))

#                 if len(non_zero_positions) == 0:
#                     # whole sequence is zero (no feature present)
#                     print(f"No feature present in the sequence for index {index}. Skipping.")
#                     continue  # skip bc no feature present

#                 elif len(non_zero_positions) == len(feature_values):
#                     # whole sequence has no zero values (feature always present to some degree)
#                     # find the most variable region
#                     window_size = 10  # adjust window size
#                     variances = [np.var(feature_values[i:i + window_size]) for i in range(len(feature_values) - window_size + 1)]

#                     # window with max variance in feature values
#                     max_var_index = np.argmax(variances)
#                     subsequence_start = max_var_index
#                     subsequence_end = subsequence_start + window_size

#                     print(f"Most variable region: {feature_values[subsequence_start:subsequence_end]}")

#                 else:
#                     # case where there are non-zero and zero values mixed (1 indicates presence)
#                     start = non_zero_positions[0]  # first non-zero position

#                     # find the first zero after the start
#                     continuous_non_zero = np.where(feature_values[start:] == 0)[0]

#                     if len(continuous_non_zero) > 0:
#                         end = start + continuous_non_zero[0]  # stop at the first zero
#                     else:
#                         end = non_zero_positions[-1] + 1  # if no zero found, take until last non-zero position

#                     feature_len = end - start
#                     flank = feature_len // 2
#                     subsequence_start = max(0, start - flank)
#                     subsequence_end = min(seq_len, end + flank)

#                     print(f"Sequence {index}: {subsequence_start} - {subsequence_end}")
#                     print(f"Feature Length: {feature_len}")

#                 # feature values and attention scores for this sequence
#                 subsequence_data = {
#                     'feature_values': feature_values[subsequence_start:subsequence_end]
#                 }

#                 # extract attention scores for all layers and heads within this range
#                 for layer_head in attention_score_columns:
#                     attention_subseq = row[layer_head][subsequence_start:subsequence_end]
#                     subsequence_data[layer_head] = attention_subseq

#                 #save the sequence identifier without compression
#                 seq_id = row[seq_column]  # seq identifier
#                 seq_dataset_name = f'{index}/seq_column'
#                 feature_group.create_dataset(seq_dataset_name, data=seq_id)  # No compression for scalar

#                 # store subsequence in the HDF5 group with compression
#                 for key, value in subsequence_data.items():
#                     feature_group.create_dataset(f'{index}/{key}', data=value, compression="gzip", compression_opts=9)

#                 # saved example counter +
#                 saved_example_count += 1

#         print(f"Saved feature-specific matrices to {hdf5_file}")


def extract_and_save_subsequences_for_feature(df, feature, seq_len, attention_score_columns, seq_column, output_folder, model_name):
    # folder for saving feature-specific matrices
    path_addition = f'{model_name}/'
    model_folder = os.path.join(output_folder, path_addition)
    os.makedirs(model_folder, exist_ok=True)
    
    hdf5_file = os.path.join(model_folder, f'feature_scores.h5')  # HDF5 file
    # lock file to synchronize access
    lock_file = hdf5_file + ".lock"
    with FileLock(lock_file):
        with h5py.File(hdf5_file, 'a') as hdf5_store:  # Use 'a' for appending data
            feature_path = feature.replace("/", "+")
            print(f"\nExtracting attention score subsequences for feature: {feature}")
            
            if feature not in df.columns:
                print(f"Warning: Feature '{feature}' not found in DataFrame")
                return

            # group for the feature in the HDF5 file
            feature_group = hdf5_store.require_group(feature_path)

            saved_example_count = 0  # Initialize counter for saved examples per feature

            # per sequence
            for index, row in df.iterrows():
                if saved_example_count >= 200:
                    print(f"Reached 200 examples for feature {feature}. Stopping further processing for this feature.")
                    break

                feature_values = np.array(row[feature])
                print(f"Feature values: {feature_values}")

                # positions where the feature is present (non-zero values)
                non_zero_positions = np.where(feature_values != 0)[0]
                print(len(non_zero_positions))

                if len(non_zero_positions) == 0:
                    # Whole sequence is zero (no feature present)
                    print(f"No feature present in the sequence for index {index}. Skipping.")
                    continue  # Skip because no feature is present

                elif len(non_zero_positions) == len(feature_values):
                    # Whole sequence has no zero values (feature always present to some degree)
                    # Find the most variable region
                    window_size = 10  # Adjust window size
                    variances = [np.var(feature_values[i:i + window_size]) for i in range(len(feature_values) - window_size + 1)]

                    # Window with max variance in feature values
                    max_var_index = np.argmax(variances)
                    subsequence_start = max_var_index
                    subsequence_end = subsequence_start + window_size

                    print(f"Most variable region: {feature_values[subsequence_start:subsequence_end]}")

                else:
                    # Case where there are non-zero and zero values mixed (1 indicates presence)
                    start = non_zero_positions[0]  # First non-zero position

                    # Find the first zero after the start
                    continuous_non_zero = np.where(feature_values[start:] == 0)[0]

                    if len(continuous_non_zero) > 0:
                        end = start + continuous_non_zero[0]  # Stop at the first zero
                    else:
                        end = non_zero_positions[-1] + 1  # If no zero found, take until last non-zero position

                    feature_len = end - start
                    flank = feature_len // 2
                    subsequence_start = max(0, start - flank)
                    subsequence_end = min(seq_len, end + flank)

                    print(f"Sequence {index}: {subsequence_start} - {subsequence_end}")
                    print(f"Feature Length: {feature_len}")

                # Store feature values and attention scores for this sequence
                subsequence_data = {
                    'feature_values': feature_values[subsequence_start:subsequence_end]
                }

                # Extract attention scores for all layers and heads within this range
                for layer_head in attention_score_columns:
                    attention_subseq = row[layer_head][subsequence_start:subsequence_end]
                    subsequence_data[layer_head] = attention_subseq

                # Save the sequence identifier without compression
                seq_id = row[seq_column]  # Seq identifier
                seq_dataset_name = f'{index}/seq_column'
                feature_group.create_dataset(seq_dataset_name, data=seq_id)  # No compression for scalar

                # Store subsequence in the HDF5 group with compression
                for key, value in subsequence_data.items():
                    feature_group.create_dataset(f'{index}/{key}', data=value, compression="gzip", compression_opts=9)

                # Saved example counter +
                saved_example_count += 1

    print(f"Saved feature-specific matrices to {hdf5_file} for feature {feature}")




#set up for different models configurations
def configure_DNABERT(df):
    #copy to modify
    df_copy = df.copy()
    for col in df_copy.columns:
        if col not in ['sequence', 'kmers']:
            df_copy[col] = df_copy[col].apply(lambda x: np.array(x.split(','), dtype=np.float64))
    return {
        'attention_score_columns': [col for col in df_copy.columns if 'layer' in col and 'head' in col],
        'bio_feature_columns': [col for col in df_copy.columns if not (('layer' in col and 'head' in col) or ('label' in col) or ('sequence' in col) or ('kmers' in col))],
        'seq_length': df_copy['kmers'].apply(lambda x: len(x.split(','))).tolist(),  # sequence lengths based on 'kmers',
        'seq_column': 'sequence',
        'transformed_df': df_copy, # transformed df
    }


def configure_enformer(df):
    # copy to modify
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if col != 'gene':
            df_copy[col] = df_copy[col].apply(lambda x: np.array(x.split(','), dtype=np.float64))
    
    return {
        'attention_score_columns': [col for col in df_copy.columns if 'layer' in col and 'head' in col],
        'bio_feature_columns': [col for col in df_copy.columns if not (('layer' in col and 'head' in col) or ('gene' in col))],
        'seq_length': 1536,
        'seq_column': 'gene',
        'transformed_df': df_copy  # transformed df
    }

def configure_scgpt(df):
    df_copy = df.copy()
    for col in df_copy.columns:
        if col not in ['gene_sequence', 'label']:
            df_copy[col] = df_copy[col].apply(lambda x: np.array(x.split(','), dtype=np.float64))
    return {
        'attention_score_columns': [col for col in df_copy.columns if 'layer' in col and 'head' in col],
        'bio_feature_columns': [col for col in df_copy.columns if not (('layer' in col and 'head' in col) or ('gene_sequence' in col) or  ('label' in col))],
        'seq_length': 500,
        'seq_column': 'gene_sequence',
        'transformed_df': df_copy # transformed df
    }

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
    
    # Configuration functions for different models
    config_functions = {
        'DNABERT': configure_DNABERT,
        'DNABERT_pretrained': configure_DNABERT,
        'DNABERT_random': configure_DNABERT,
        'DNABERT_random_init': configure_DNABERT,
        'DNABERT_TATA': configure_DNABERT,
        'enformer': configure_enformer,
        'enformer_random_init': configure_enformer
    }

    # Default to scGPT configurations if model not found
    config_function = config_functions.get(args.model_name, configure_scgpt)
    
    # Apply configuration for the run
    config = config_function(df)

    new_df = config["transformed_df"] if "transformed_df" in config else df

    def check_types(df):
        for col in df.columns:
            col_types = df[col].apply(lambda x: type(x).__name__)
            print(f"Column '{col}' contains types: {col_types.unique()}")
    # read data

    check_types(new_df)
    
    attention_score_columns = config["attention_score_columns"]
    bio_feature_columns = config["bio_feature_columns"]
    seq_length = config["seq_length"]
    seq_column = config['seq_column']

    # do not include biological features containing the word 'position'
    bio_feature_columns = [col for col in bio_feature_columns if 'position' not in col]


    print(f'Attention Score Columns: {attention_score_columns}')
    print(f'Bio Feature Columns: {bio_feature_columns}')
    print(f'Sequence Length: {seq_length}')

    output_folder = f'{full_path}/data/feature_scores/'
    os.makedirs(output_folder, exist_ok=True)

    # Check for missing features
    missing_features = check_existing_files(full_path, model, bio_feature_columns)

    print(f"CHECKED MISSING FEATURES:{missing_features}")

    # if missing_features:
    #     # Get subsequences per bio feature
    #     extract_and_save_subsequences(new_df, bio_feature_columns, seq_length, attention_score_columns, seq_column, output_folder, model)
    # else:
    #     print("All features already processed.")
    if missing_features:
        # Parallelize feature processing
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(extract_and_save_subsequences_for_feature, new_df, feature, seq_length, attention_score_columns, seq_column, output_folder, model)
                for feature in missing_features
            ]

            for future in futures:
                future.result()  # Ensures exceptions are raised if any

    else:
        print("All features already processed.")
    # sequence IDs to identifiers
    seq_id_mapping = map_sequence_ids_to_identifiers(f'{full_path}/data/feature_scores/{model}/feature_scores.h5')
    #save seq_id mapping to a csv file
    seq_id_mapping_df = pd.DataFrame(seq_id_mapping.items(), columns=['seq_id', 'seq_identifier'])
    seq_id_mapping_df.to_csv(f'{full_path}/data/feature_scores/{model}/seq_id_mapping.csv', index=False)

if __name__ == "__main__":
    main()
