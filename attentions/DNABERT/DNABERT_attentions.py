import torch
import torch.nn.functional as F
import sys
from transformers import BertConfig, DNATokenizer, BertForSequenceClassification
from transformers import BertForMaskedLM
from torch.utils.data import Dataset, DataLoader
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
torch.cuda.empty_cache()


def verify_initialization_basic(model: nn.Module, mean: float = 0.0, std: float = 0.02, tolerance: float = 0.01):
    """
    A simpler, less memory-intensive verification function that doesn't create plots
    """
    # Count total parameters and layers
    total_params = 0
    total_correct_params = 0
    total_layers = 0
    total_correct_layers = 0
    
    # Count layer types
    layer_counts = {"linear": 0, "embedding": 0, "layernorm": 0, "other": 0}
    
    # Check each layer
    all_correct = True
    issues = []
    
    print("\nVerifying initialization against standardized scheme:")
    print(f"- Linear layers: weights ~ N({mean}, {std}), biases = 0")
    print(f"- Embedding layers: weights ~ N({mean}, {std})")
    print(f"- LayerNorm: gamma = 1.0, beta = 0.0")
    print(f"- Other weight parameters: ~ N({mean}, {std})")
    print("-" * 50)
    
    for name, module in model.named_modules():
        # Skip container modules
        if len(list(module.parameters(recurse=False))) == 0:
            continue
        
        layer_correct = True
        
        # Check by module type
        if isinstance(module, nn.Linear):
            layer_counts["linear"] += 1
            
            # Check weights
            weights = module.weight.detach().cpu().numpy().flatten()
            w_mean = np.mean(weights)
            w_std = np.std(weights)
            
            weights_correct = (abs(w_mean - mean) < tolerance and 
                              abs(w_std - std) < tolerance * 3)
            
            # Check biases
            bias_correct = True
            if module.bias is not None:
                biases = module.bias.detach().cpu().numpy().flatten()
                bias_mean = np.mean(np.abs(biases))
                bias_correct = bias_mean < tolerance / 10
            
            layer_correct = weights_correct and bias_correct
            
            total_params += len(weights)
            if weights_correct:
                total_correct_params += len(weights)
            
            if not layer_correct:
                issues.append({
                    "layer": name,
                    "type": "linear",
                    "w_mean": w_mean,
                    "w_std": w_std,
                    "bias_correct": bias_correct
                })
                all_correct = False
                
        elif isinstance(module, nn.Embedding):
            layer_counts["embedding"] += 1
            
            # Check weights
            weights = module.weight.detach().cpu().numpy().flatten()
            w_mean = np.mean(weights)
            w_std = np.std(weights)
            
            weights_correct = (abs(w_mean - mean) < tolerance and 
                              abs(w_std - std) < tolerance * 3)
            
            layer_correct = weights_correct
            
            total_params += len(weights)
            if weights_correct:
                total_correct_params += len(weights)
            
            if not layer_correct:
                issues.append({
                    "layer": name,
                    "type": "embedding",
                    "w_mean": w_mean,
                    "w_std": w_std
                })
                all_correct = False
                
        elif isinstance(module, nn.LayerNorm):
            layer_counts["layernorm"] += 1
            
            # Check gamma and beta
            gamma = module.weight.detach().cpu().numpy().flatten()
            beta = module.bias.detach().cpu().numpy().flatten()
            
            gamma_correct = np.allclose(gamma, 1.0, atol=tolerance)
            beta_correct = np.allclose(beta, 0.0, atol=tolerance)
            
            layer_correct = gamma_correct and beta_correct
            
            total_params += len(gamma) + len(beta)
            if layer_correct:
                total_correct_params += len(gamma) + len(beta)
            
            if not layer_correct:
                issues.append({
                    "layer": name,
                    "type": "layernorm",
                    "gamma_mean": np.mean(gamma),
                    "beta_mean": np.mean(beta)
                })
                all_correct = False
                
        else:
            # Handle other layer types
            layer_counts["other"] += 1
            
            for param_name, param in module.named_parameters(recurse=False):
                if 'weight' in param_name:
                    weights = param.detach().cpu().numpy().flatten()
                    w_mean = np.mean(weights)
                    w_std = np.std(weights)
                    
                    weights_correct = (abs(w_mean - mean) < tolerance and 
                                      abs(w_std - std) < tolerance * 3)
                    
                    total_params += len(weights)
                    if weights_correct:
                        total_correct_params += len(weights)
                    
                    if not weights_correct:
                        issues.append({
                            "layer": f"{name}.{param_name}",
                            "type": "other_weight",
                            "w_mean": w_mean,
                            "w_std": w_std
                        })
                        layer_correct = False
                        all_correct = False
                        
                elif 'bias' in param_name:
                    biases = param.detach().cpu().numpy().flatten()
                    bias_mean = np.mean(np.abs(biases))
                    bias_correct = bias_mean < tolerance / 10
                    
                    total_params += len(biases)
                    if bias_correct:
                        total_correct_params += len(biases)
                    
                    if not bias_correct:
                        issues.append({
                            "layer": f"{name}.{param_name}",
                            "type": "other_bias",
                            "bias_mean": bias_mean
                        })
                        layer_correct = False
                        all_correct = False
        
        # Count correct layers
        total_layers += 1
        if layer_correct:
            total_correct_layers += 1
    
    # Print summary
    print("\nVerification Summary:")
    print(f"- Linear layers: {layer_counts['linear']}")
    print(f"- Embedding layers: {layer_counts['embedding']}")
    print(f"- LayerNorm layers: {layer_counts['layernorm']}")
    print(f"- Other layers: {layer_counts['other']}")
    print(f"- Total layers checked: {total_layers}")
    print(f"- Correctly initialized layers: {total_correct_layers}/{total_layers} ({100*total_correct_layers/total_layers:.1f}%)")
    print(f"- Correctly initialized parameters: {total_correct_params}/{total_params} ({100*total_correct_params/total_params:.1f}%)")
    
    # Show issues if any
    if not all_correct:
        print(f"\n⚠️ Found {len(issues)} issues with initialization:")
        for i, issue in enumerate(issues[:5]):  # Show top 5 issues
            print(f"  {i+1}. {issue['layer']} ({issue['type']}): ", end="")
            if 'gamma_mean' in issue:
                print(f"gamma={issue['gamma_mean']:.4f} (expected: 1.0), "
                      f"beta={issue['beta_mean']:.4f} (expected: 0.0)")
            elif 'bias_mean' in issue:
                print(f"bias_mean={issue['bias_mean']:.6f} (expected: 0.0)")
            else:
                print(f"mean={issue['w_mean']:.6f} (expected: {mean}), "
                      f"std={issue['w_std']:.6f} (expected: {std})")
        if len(issues) > 5:
            print(f"  ... and {len(issues)-5} more issues.")
    else:
        print("\n✅ All layers are correctly initialized according to the standardized scheme!")
    
    return all_correct

def initialize_weights_normal(model: nn.Module, mean: float = 0.0, std: float = 0.02):
    """
    Initialize all model weights using a normal distribution N(mean, std).
    
    Args:
        model: PyTorch model
        mean: Mean of the normal distribution (default: 0.0)
        std: Standard deviation of the normal distribution (default: 0.02)
    """
    for module in model.modules():
        # Linear layers
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        
        # Embedding layers
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        # LayerNorm
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)  # gamma
            module.bias.data.zero_()       # beta
        
        # Conv layers (in case there are any)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        
        # Any other layer with parameters
        else:
            for param_name, param in module.named_parameters(recurse=False):
                if 'weight' in param_name:
                    # Apply normal initialization to any other weight parameters
                    param.data.normal_(mean=mean, std=std)
                elif 'bias' in param_name:
                    # Zero initialization for any bias parameters
                    param.data.zero_()

def random_init_dnabert(model, mean=0.0, std=0.02):
    """
    Random initialization for DNABERT model.
    
    Args:
        model: DNABERT model (typically BertForMaskedLM)
        mean: Mean of normal distribution
        std: Standard deviation of normal distribution
    
    Returns:
        The randomly initialized model
    """
    print("Applying standardized random initialization to DNABERT model...")
    initialize_weights_normal(model, mean=mean, std=std)
    return model


class DNADataset(Dataset):
    def __init__(self, inputs, tokenizer, max_seq_length=200):
        self.inputs = inputs
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.inputs)

    def __seq2kmer__(self, seq, k):
        """
        Convert original sequence to kmers
        
        Arguments:
        seq -- str, original sequence.
        k -- int, kmer of length k specified.
        
        Returns:
        kmers -- str, kmers separated by space

        """
        kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
        kmers = " ".join(kmer)
        return kmers

    def __getitem__(self, idx):
        row = self.inputs.iloc[idx]
        seq = row['sequence']
        label = row['label']

        max_seq_length =510
        encoding = self.tokenizer.encode_plus(seq,  sentence_b=None, add_special_tokens=True, max_length=max_seq_length)
        input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
        pad_token_segment_id = 0
        padding_length = max_seq_length - len(input_ids)
        # mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [0] * len(input_ids)
        pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        input_ids = input_ids + ([pad_token] * padding_length)
        #mask is -inf not 0
        attention_mask = attention_mask + ([-np.inf] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        inputs =({
                    "input_ids": torch.Tensor(input_ids).long(),
                    "attention_mask": torch.Tensor(attention_mask),
                    "token_type_ids": torch.Tensor(token_type_ids).long(),
                })

        return inputs



def create_data_loader(inputs, tokenizer):
    dataset = DNADataset(inputs, tokenizer)
    data_loader = DataLoader(dataset, batch_size=4, pin_memory=True, num_workers=0)  # adjust as necessary
    return data_loader


def get_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        return {line.strip(): i for i, line in enumerate(f)}


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
def analyze_attention_heads(model, data_loader, device, tokenizer, vocab):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    model.to(device)
    model.eval()


    #dictionaries to store all examples and their scores for each head
    examples_scores_attention = {layer: {head: [] for head in range(num_heads)} for layer in range(num_layers)}

    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids', None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        all_attentions = outputs[-1]  # assuming output_attentions=True


        for layer in range(num_layers):
            for head in range(num_heads):
                attention_scores_per_head = all_attentions[layer][:, head, :, :].detach()
                
                attention_scores = attention_scores_per_head
                
                input_tokens = [tokenizer.convert_ids_to_tokens(input_id[1:-1]) for input_id in input_ids.tolist()]

                for tokens, att_matrix in zip(input_tokens, attention_scores):
                    # row-wise max score
                    max_att_scores = att_matrix.max(dim=0)[0].detach().cpu().numpy()  
                    examples_scores_attention[layer][head].append((max_att_scores, tokens))
                  
    return examples_scores_attention

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/scratch/ssd004/scratch/mconsens/examples/ft/6/full_enhancer/",
        type=str,
        help="The path of the finetuned model",
    )
    parser.add_argument(
        "--data_path",
        default="/scratch/ssd004/scratch/mconsens/examples/sample_data/ft/6/full_enhancer/dev.tsv",
        type=str,
        help="The path to the data",
    )
    parser.add_argument(
        "--task_name",
        default="human_enhancer",
        type=str,
        help="The task name, either human_enhancer or human_non_TATA or pretrained or random/random_init",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/DNABERT/Transformer-Explainability",
        type=str,
        help="The full path to the attention-head-analysis.py file",
    )
    parser.add_argument(
        "--num_samples",
        default=400,
        type=int,
        help="Number of samples from data",
    )
    args = parser.parse_args()

    model_path = args.model_path
    data_path = args.data_path
    task_name = args.task_name
    full_path = args.full_path
    num_samples = args.num_samples
    
    config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, DNATokenizer

    #model
    model = model_class.from_pretrained(model_path, output_attentions=True)
    model.eval()
    if task_name =='random_init' or task_name=='TATA_random_init' or task_name=='fake_TATA_random_init':
        #random initialization
        model = random_init_dnabert(model)
        verify_initialization_basic(model)
    
    #tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path, do_lower_case=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    #load 
    full_dataset = pd.read_csv(args.data_path, sep='\t', header=0)
    
    # separate by label to balance dataset
    label_0_samples = full_dataset[full_dataset['label'] == 0].sample(n=num_samples, random_state=1)
    label_1_samples = full_dataset[full_dataset['label'] == 1].sample(n=num_samples, random_state=1)
    
    #concat samples from each label to form a balanced dataset
    balanced_dataset = pd.concat([label_0_samples, label_1_samples])

    #data loader
    data_loader = create_data_loader(balanced_dataset, tokenizer)

    #attention and LRP heads
    vocab = get_vocab(f'{args.model_path}/vocab.txt')
    results = analyze_attention_heads(model, data_loader, device, tokenizer, vocab)

    # unpack results
    examples_scores_attention = results

    #save scores in layer-indexed-files
    os.makedirs(f'{args.full_path}/attention/{args.task_name}/', exist_ok=True)
    for layer in range(model.config.num_hidden_layers):
        #examples_scores_attention for the layer
        attention_filename = f'{args.full_path}/attention/{args.task_name}/examples_scores_attention_{num_samples}_layer{layer}.p'
        with open(attention_filename, 'wb') as f:
            pickle.dump(examples_scores_attention[layer], f)

if __name__ == "__main__":
    main()
