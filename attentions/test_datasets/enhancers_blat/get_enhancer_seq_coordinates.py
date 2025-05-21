import os
import pickle
import pandas as pd
from pyfaidx import Fasta
import subprocess

def kmer2seq(kmers_list:list) -> str:
    first:bool = True
    for token in kmers_list:
        if first:
            sequence:str = token
            first = False
        elif (token != '[SEP]' and token != '[PAD]'):
            sequence += token[-1]
        else:
            break
    return(sequence)

def print_results(results) -> pd.DataFrame:
    df:pd.DataFrame = pd.DataFrame(columns=["Query", "Score", "Chrom", "Start", "End", "Strand", "SeqLen", "Match", "SPAN"])
    for idx, query in enumerate(results):
        for hit in query.hsps:
            try:
                df.loc[len(df)] = [idx, hit.score, hit.hit_id, hit.hit_start, hit.hit_end, hit.query_strand, query.seq_len, hit.match_num, hit.hit_span]
            except ValueError:
                df.loc[len(df)] = [idx, hit.score, hit.hit_id, hit.hit_start, hit.hit_end, hit[0].query_strand, query.seq_len, hit.match_num, hit.hit_span]
    return(df.sort_values(by=['Query', 'Score'], ascending=[True, False]))

def fasta2df(fasta_path):
    fasta = Fasta(fasta_path)
    data = [{"Name": name, "Sequence": str(fasta[name][:])} for name in fasta.keys()]
    data = pd.DataFrame(data)
    data["Label"] = data["Name"].str.split("|").str[-1]
    data = data.drop(columns = "Name")
    return data

# Define directories and files
base_dir = "test_datasets/enhancers_blat"
split_sequences_dir = os.path.join(base_dir, "split_sequences")
split_blat_results_dir = os.path.join(base_dir, "split_blat_results")
input_pickle = f'DNABERT/attention_scores/finetuned/{layer}0.p'
sequences_fasta = 'test_datasets/enhancers_blat/selected_enhancer_test.fa'

# Ensure directories exist
os.makedirs(split_sequences_dir, exist_ok=True)
os.makedirs(split_blat_results_dir, exist_ok=True)

# Load the results from the pickle file
sequences_list = []
with open(input_pickle, 'rb') as f:
    results: dict = pickle.load(f)

for example in range(len(results[0])):
    sequence:str = kmer2seq(results[0][example][1])
    sequences_list.append(sequence)

# Save the sequences into several files to run Blat in chunks
chunk_size = 200
for i in range(0, len(sequences_list), chunk_size):
    chunk_filename = os.path.join(split_sequences_dir, f"sequences_{i // chunk_size}")
    with open(chunk_filename, "w") as chunk_file:
        chunk_file.write('\n'.join(sequences_list[i:(i+chunk_size)]))

# Run `blat_dnabert.py` on each chunk
for chunk_file in sorted(os.listdir(split_sequences_dir)):
    input_path = os.path.join(split_sequences_dir, chunk_file)
    output_path = os.path.join(split_blat_results_dir, f"{chunk_file}_blat.pkl")
    subprocess.run(["python", "blat.py", "--input", input_path, "-o", output_path])

# Read all BLAT pickle files and save the results into a dataframe
blat_query_seq_results:pd.DataFrame = pd.DataFrame()
for idx, pkl in enumerate(sorted(os.listdir(split_blat_results_dir))):
    with open(os.path.join(split_blat_results_dir, pkl), 'rb') as f:
        tmp = pickle.load(f)
    tmp = print_results(tmp)
    tmp['Query'] = tmp['Query'] + (idx*chunk_size) #Each file contains nchunk sequences
    blat_query_seq_results = pd.concat([blat_query_seq_results, tmp])
blat_query_seq_results.reset_index(drop=True, inplace=True)

# Filter results
blat_query_seq_results_filtered:pd.DataFrame = blat_query_seq_results[
    ((blat_query_seq_results['End']-blat_query_seq_results['Start']) == blat_query_seq_results['Match']) &
    ((blat_query_seq_results['End']-blat_query_seq_results['Start']) == blat_query_seq_results['Score']) &
    ((blat_query_seq_results['End']-blat_query_seq_results['Start']) == blat_query_seq_results['SPAN']) &
    (blat_query_seq_results['Chrom'].isin(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']))
]
blat_query_seq_results_filtered.reset_index(drop=True, inplace=True)

# Remove repetitive queries, they contain the same sequence information
blat_query_seq_results_filtered = blat_query_seq_results_filtered.drop_duplicates(subset=['Query'])
blat_query_seq_results_filtered.reset_index(drop=True, inplace=True)

# Append sequence to the dataframe
blat_query_seq_results_filtered['Sequence'] = [sequences_list[idx] for idx in blat_query_seq_results_filtered['Query']]

# Append label to the dataframe
fasta_df = fasta2df(sequences_fasta)
blat_query_seq_results_filtered = pd.merge(blat_query_seq_results_filtered, fasta_df,
                                           on="Sequence",
                                           how="left")
blat_query_seq_results_filtered.to_csv('../enhancer_test.csv', index=False)