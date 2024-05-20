import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'  # or any other value like 512
import torch
from enformer_pytorch import Enformer 
import cProfile
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from torch.cuda.amp import GradScaler, autocast

torch.cuda.empty_cache()


class DNADataset(Dataset):
    def __init__(self, csv_file, num_samples=None, max_seq_length=196_608):
        self.dataframe = pd.read_csv(csv_file)
        if num_samples is not None and 0 < num_samples < len(self.dataframe):
            self.dataframe = self.dataframe.sample(n=num_samples, random_state=42).reset_index(drop=True)
        self.max_seq_length = max_seq_length

    @staticmethod
    def one_hot_encode(sequence, max_seq_length=196_608):
        base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        one_hot = torch.zeros(max_seq_length, 5, dtype=torch.float32)
        indices = torch.tensor([base_to_index.get(base, 4) for base in sequence], dtype=torch.long)
        valid_length = min(max_seq_length, len(indices))
        valid_indices = indices[:valid_length]
        rows = torch.arange(valid_length)
        one_hot[rows, valid_indices] = 1
        one_hot = one_hot[:, :4]
        return one_hot


    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sequence = row['sequence']
        label = row['label']
        encoded_sequence = DNADataset.one_hot_encode(sequence, self.max_seq_length)
        seq_len = min(len(sequence), self.max_seq_length)
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.float32)
        attention_mask[:seq_len] = 1
        inputs = {
            "input_ids": encoded_sequence,
            "label": label,
            "attention_mask": attention_mask,
        }
        return inputs

def create_data_loader(csv_file, num_samples=None, max_seq_length=196_608, batch_size=2, num_workers=1):
    dataset = DNADataset(csv_file, num_samples, max_seq_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return data_loader



'''
Data Organization:

    examples_scores_attention: 
    Nested dictionaries where the first level of keys represents the layer number (0 to num_layers - 1), and the second level of keys 
    represents the head number (0 to num_heads - 1). The value for each (layer, head) key is a list of tuples of length equal to the 
    number of examples processed, each tuple consisting of the attentionscore list  for each token in an example 
    (where the attention score is the row-wise max of scores per token from the attention matrix)
    and the list of input tokens for that example.

    Layer (key1)
        Head (key2)
            Example Tuple (example1, example2, ..., exampleN)
                Average Attention Scores per Token (value1 of the tuple for exampleN)
                    A list containing the max attention score for each token in the example, calculated row-wise.
                Tokens (value2 of the tuple for exampleN)
                    A list of tokens corresponding to the input sequence of the example. 
                    Each token corresponds to a score in the list of  attention scores (value1 of the tuple for exampleN).

'''
def analyze_attention_heads(model, data_loader, device, full_path):
    num_layers = model.config.depth
    num_heads = model.config.heads

    model.to(device)
    model.eval()

    examples_scores_attention = {layer: {head: [] for head in range(num_heads)} for layer in range(num_layers)}

    for batch_num, batch in enumerate(data_loader):
        print(f"BATCH NUM: {batch_num}")
        input_ids = batch['input_ids'].to(device).half()
        attention_mask = batch['attention_mask'].to(device).half()

        with torch.no_grad():  # no gradients are computed to save memory
            outputs = model(input_ids)
            all_attentions = outputs[1]  # attentions
      

        for layer in range(num_layers):
            for head in range(num_heads):
                attention_scores = all_attentions[layer][:, head, :, :]
               
                #label the scores by the number of nucleotides (nucleotide bins) they represent
                input_tokens = [x * 128 for x in range(attention_scores.shape[1])]
              
                for att_matrix in attention_scores:
                   #rowwise max score
                    max_att_scores = att_matrix.max(dim=0)[0].detach().cpu().numpy() 
                    examples_scores_attention[layer][head].append((max_att_scores, input_tokens))

        # clear variables and empty cache after each batch to save memory
        del all_attentions
        torch.cuda.empty_cache()
    print("Analysis completed.")
    return examples_scores_attention #, examples_scores_lrp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/scratch/ssd004/scratch/mconsens/Enformer/enformer-pytorch/model/pytorch_model.bin",
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/Enformer/enformer-pytorch/seq_data/output_sequences.csv",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/Enformer/enformer-pytorch/",
        type=str,
        help="The full path to the attention-head-top-examples.py file",
    )
    parser.add_argument(
        "--num_samples",
        default=200,
        type=int,
        help="Number of samples to process",
    )
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    full_path = args.full_path
    num_samples = args.num_samples
    
    
    #model 
    model = Enformer.from_pretrained(model_path)
    model.eval()
    model.output_attentions = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    SEQUENCE_LENGTH = 196_608
    TARGET_LENGTH = 896

    #data loader
    data_loader = create_data_loader(data_path, num_samples=num_samples, batch_size=1, num_workers=1)

    examples_scores_attention = analyze_attention_heads(model, data_loader, device, full_path)
    
    num_layers = model.config.depth
    num_heads = model.config.heads

    # directory exists
    os.makedirs(f'{full_path}/attention/', exist_ok=True)
    #save scores in layer-indexed-files
    for layer in range(num_layers):
        #examples_scores_attention for the layer
        attention_filename = f'{full_path}/attention/examples_scores_attention_layer{layer}.p'
        with open(attention_filename, 'wb') as f:
            pickle.dump(examples_scores_attention[layer], f)
      
if __name__ == "__main__":
    main()

