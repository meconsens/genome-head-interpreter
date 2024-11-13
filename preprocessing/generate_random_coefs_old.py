import h5py
import random
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from filelock import FileLock

def get_bio_and_attention_columns_from_hdf5(hdf5_file_path):
    with h5py.File(hdf5_file_path, 'r') as hdf5_store:
        # Biological features are the top-level groups
        bio_feature_columns = list(hdf5_store.keys())

        # To extract attention heads, check any sequence under any bio feature
        first_bio_feature = bio_feature_columns[0]  # Take the first biological feature group
        first_seq_id = list(hdf5_store[first_bio_feature].keys())[0]  # Take the first sequence ID under this bio feature

        # Extract the keys (datasets) under the first sequence for attention scores
        first_seq_group = hdf5_store[first_bio_feature][first_seq_id]
        attention_heads = [key for key in first_seq_group.keys() if key.startswith('layer')]

    return bio_feature_columns, attention_heads


# Sample random regions from HDF5 for a single feature
def sample_random_regions_for_feature(hdf5_file_path, bio_feature, attention_heads, output_hdf5_file, seq_id, num_random_samples=100):
    with h5py.File(hdf5_file_path, 'r') as hdf5_store:
        feature_group = hdf5_store[bio_feature]
        
        # Retrieve actual data for the feature and sequence
        actual_feature_data = {}
        actual_seq_len = len(feature_group[seq_id]['feature_values'][:])
        
        for attention_head in attention_heads:
            actual_feature_data[attention_head] = feature_group[seq_id][attention_head][:]  # Extract attention scores
        actual_feature_data['feature_values'] = feature_group[seq_id]['feature_values'][:]  # Extract feature values

        # Sample random regions from other sequences
        random_samples = []
        all_seq_ids = list(feature_group.keys())
        all_seq_ids.remove(seq_id)  # Exclude current sequence

        for _ in range(num_random_samples):
            random_seq_id = random.choice(all_seq_ids)  # Pick a random sequence ID
            random_feature_values = feature_group[random_seq_id]['feature_values'][:]
            
            if len(random_feature_values) >= actual_seq_len:
                random_start = random.randint(0, len(random_feature_values) - actual_seq_len)
                random_end = random_start + actual_seq_len

                random_sample = {
                    'random_feature_values': random_feature_values[random_start:random_end]
                }

                for attention_head in attention_heads:
                    random_sample[attention_head] = feature_group[random_seq_id][attention_head][random_start:random_end]

                random_samples.append(random_sample)

        save_random_samples_to_hdf5(output_hdf5_file, bio_feature, seq_id, actual_feature_data, random_samples)

# Save random samples to HDF5
def save_random_samples_to_hdf5(output_hdf5_file, feature_name, seq_id, actual_data, random_samples):
    lock_file = output_hdf5_file + ".lock"
    with FileLock(lock_file):
        with h5py.File(output_hdf5_file, 'a') as hdf5_store:
            feature_group = hdf5_store.require_group(feature_name)
            seq_group = feature_group.create_group(seq_id)

            # Save actual data with compression for array-like data
            for key, value in actual_data.items():
                if np.isscalar(value):  # No compression for scalar data
                    seq_group.create_dataset(f'actual_{key}', data=value)
                else:
                    seq_group.create_dataset(f'actual_{key}', data=value, compression="gzip", compression_opts=9)

            # Save random samples with compression for array-like data
            for i, sample in enumerate(random_samples):
                sample_group = seq_group.create_group(f'random_sample_{i}')
                for key, value in sample.items():
                    if np.isscalar(value):  # No compression for scalar data
                        sample_group.create_dataset(key, data=value)
                    else:
                        sample_group.create_dataset(key, data=value, compression="gzip", compression_opts=9)


# parallel for features
def parallel_sample_for_feature(hdf5_file_path, bio_feature, attention_heads, seq_ids, output_hdf5_file, num_random_samples=100):
    feature_name = bio_feature.replace("/", "+")
    for seq_id in seq_ids:
        # Sample random regions for the current feature and sequence ID
        sample_random_regions_for_feature(
            hdf5_file_path, 
            feature_name,  # Pass individual feature name here
            attention_heads, 
            output_hdf5_file, 
            seq_id, 
            num_random_samples
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/scores/DNABERT_scores.csv",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/",
        type=str,
        help="The full path to save shuffled matrices and results",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT",
        type=str,
        help="The model being explained",
    )
    args = parser.parse_args()

    model = args.model_name

    # Paths for input and output files
    hdf5_input_path = args.data_path
    output_hdf5_file = f"{args.full_path}/data/distributions/{model}/random_samples.h5"
    
    # Extract bio feature columns and attention score columns from HDF5
    bio_features, attention_heads = get_bio_and_attention_columns_from_hdf5(hdf5_input_path)

    print(f"Bio Feature Columns: {bio_features}")
    print(f"Attention Score Columns: {attention_heads}")

    # sequence ids
    with h5py.File(hdf5_input_path, 'r') as hdf5_store:
        seq_ids = list(hdf5_store[bio_features[0]].keys())  # Use any feature to get sequence ids

    num_random_samples=2

    # parallel processing for bio features
    with ProcessPoolExecutor(max_workers=100) as executor:  # Adjust max_workers based on the system
        futures = []
        for bio_feature in bio_features:
            futures.append(executor.submit(parallel_sample_for_feature, hdf5_input_path, bio_feature, attention_heads, seq_ids, output_hdf5_file, num_random_samples))

        for future in as_completed(futures):
            future.result()  # Ensures exceptions are raised if any

    print(f"Random samples saved to {output_hdf5_file}")

if __name__ == "__main__":
    main()


#hdf5 input file:/ (Root)
# ├── phyloP (biological feature group)
# │   ├── seq_id_1 (sequence ID)
# │   │   ├── layer0-head0 (attention head data)
# │   │   ├── layer0-head1 (attention head data)
# │   │   └── feature_values (biological feature values)
# │   └── seq_id_2
# │       ├── layer0-head0
# │       ├── layer0-head1
# │       └── feature_values
# ├── repeat_Retroposon (another biological feature group)
# │   ├── seq_id_1
# │   │   ├── layer0-head0
# │   │   ├── layer0-head1
# │   │   └── feature_values
# │   └── seq_id_2
# │       ├── layer0-head0
# │       ├── layer0-head1
# │       └── feature_values


#hdf5 output file:/ (Root)
# ├── phyloP (biological feature group)
# │   ├── seq_id_1 (sequence ID)
# │   │   ├── actual_feature_values (actual feature values)
# │   │   ├── actual_layer0-head0 (actual attention scores for layer0-head0)
# │   │   ├── actual_layer0-head1 (actual attention scores for layer0-head1)
# │   │   ├── random_sample_0 (first random sample)
# │   │   │   ├── random_feature_values (random feature values)
# │   │   │   ├── layer0-head0 (random attention scores for layer0-head0)
# │   │   │   └── layer0-head1 (random attention scores for layer0-head1)
# │   │   ├── random_sample_1 (second random sample)
# │   │   │   ├── random_feature_values (random feature values)
# │   │   │   ├── layer0-head0
# │   │   │   └── layer0-head1
# │   │   └── random_sample_N (Nth random sample)
# ├── repeat_Retroposon (another biological feature group)
# │   ├── seq_id_1s
# │   │   ├── actual_feature_values
# │   │   ├── actual_layer0-head0
# │   │   ├── actual_layer0-head1
# │   │   ├── random_sample_0
# │   │   │   ├── random_feature_values
# │   │   │   ├── layer0-head0
# │   │   │   └── layer0-head1
# │   │   ├── random_sample_1
# │   │   │   ├── random_feature_values
# │   │   │   ├── layer0-head0
# │   │   │   └── layer0-head1
# │   │   └── random_sample_N
