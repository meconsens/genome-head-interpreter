import pandas as pd
import math
from pyfaidx import Fasta

# Files
## Reference genome
fasta = Fasta('/databases/hg38/hg38.fa')
## Selected genes
genes = pd.read_csv('/Enformer/enformer_sequences/K562_n200_random_genes_lonely_197000.txt', header=None, names=['gene_id'])
## Coordinates
coord = pd.read_csv('/Enformer/enformer_sequences/K562_expressed_genes_lonely_197000.tsv', sep=' ')

# Process the files
## Select desire columns
coord = coord[['gene_name', 'gene_id', 'chr', 'start', 'end']]
## Keep only selected genes
df = pd.merge(left=genes, right=coord, how='left', on='gene_id')
## Remove chr prefix
df['chr'] = df['chr'].apply(lambda x: x[3:])
## Update start and end to have a length of 196608
df['new_start'] = df.apply(lambda x: x['start']-math.floor((196608-(x['end']-x['start']))/2), axis=1)
df['new_end'] = df.apply(lambda x: x['end']+math.ceil((196608-(x['end']-x['start']))/2), axis=1)

# Extract the sequences
with open('/Enformer/enformer_sequences/K562_n200_random_genes_lonely_197000.fa', 'w') as f:
    for idx, row in df.iterrows():
        f.write(f">{row['gene_id']}_{row['gene_name']}\n")
        f.write(f"{fasta[str(row['chr'])][row['new_start']:row['new_end']]}\n")

# Prepare a csv file for the posterior analysis
with open('/Enformer/enformer_sequences/sequences.csv', 'w') as f:
    f.write("label,region,sequence\n")
    for idx, row in df.iterrows():
        f.write(f"{row['gene_id']}_{row['gene_name']},{row['chr']}:{row['new_start']}-{row['new_end']},{fasta[str(row['chr'])][row['new_start']:row['new_end']]}\n")
