
import numpy as np

#set up for different models configurations
def configure_DNABERT(df):
    #copy to modify
    df_copy = df.copy()
    #print all the column names
    print(df_copy.columns)
    for col in df_copy.columns:
        if col == 'label':
                #just grab the first label and turn it into a single scalar not array
                df_copy[col] = df_copy[col].apply(lambda x: x[0])
        if col not in ['sequence', 'kmers', 'label']:
            # Check if the column contains string data before attempting to split
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].apply(lambda x: np.array(x.split(','), dtype=np.float64) if isinstance(x, str) else np.array([x], dtype=np.float64))
            else:
                # For non-string data, convert directly to numpy arrays
                df_copy[col] = df_copy[col].apply(lambda x: np.array([x], dtype=np.float64))
    return {
        'attention_score_columns': [col for col in df_copy.columns if 'layer' in col and 'head' in col],
        'bio_feature_columns': [col for col in df_copy.columns if not (('layer' in col and 'head' in col) or ('label' in col) or ('sequence' in col) or ('kmers' in col))],
        'seq_length': df_copy['kmers'].apply(lambda x: len(x.split(','))).tolist(),  # sequence lengths based on 'kmers',
        'seq_column': 'sequence',
        'transformed_df': df_copy, # transformed df
    }

def configure_nucleotide_transformer(df):
    #copy to modify
    df_copy = df.copy()
    #print all the column names
    print(df_copy.columns)
    for col in df_copy.columns:
        if col == 'label':
                #just grab the first label and turn it into a single scalar not array
                df_copy[col] = df_copy[col].apply(lambda x: x[0])
        if col not in ['sequence', 'kmers', 'label']:
            # Check if the column contains string data before attempting to split
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].apply(lambda x: np.array(x.split(','), dtype=np.float64) if isinstance(x, str) else np.array([x], dtype=np.float64))
            else:
                # For non-string data, convert directly to numpy arrays
                df_copy[col] = df_copy[col].apply(lambda x: np.array([x], dtype=np.float64))
    return {
        'attention_score_columns': [col for col in df_copy.columns if 'layer' in col and 'head' in col],
        'bio_feature_columns': [col for col in df_copy.columns if not (('layer' in col and 'head' in col) or ('label' in col) or ('sequence' in col) or ('kmers' in col))],
        'seq_length': df_copy['kmers'].apply(lambda x: len(x.split(','))).tolist(),  # sequence lengths based on 'kmers',
        'seq_column': 'sequence',
        'transformed_df': df_copy, # transformed df
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

#export all functions
__all__ = ['configure_DNABERT', 'configure_enformer', 'configure_scgpt']