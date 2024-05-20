import pandas as pd
import csv
import numpy as np
import argparse
import json
from openai import OpenAI
import dotenv
from dotenv import load_dotenv
import os
import random

#load .env variables
load_dotenv()

# API key from .env
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# coefficients into feature<tab>coef and then zero-filtered out coefs
def format_coefficients(head_data):
    # only non-zero coefficients
    formatted = "Coefficients:\n<start>\n"
    for sentence in head_data["sentences"]:
        for item in sentence:
            if isinstance(item, list) and item[1] != 0:
                formatted += f"{item[0]}\t{item[1]}\n"
    
    formatted += "<end>\n"
    formatted += f'Head Name: {head_data["given_name"]}\n'
    return formatted



def prompt_gpt(prompt):
    client = OpenAI()
    results = []

    stream = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            results.append(chunk.choices[0].delta.content)

    # join the results into a single string
    return ''.join(results)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--researcher_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/researcher_explanations/DNABERT.json",
        type=str,
        help="The path to the researcher written explanations",
    )
    parser.add_argument(
        "--explanation_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/explanation_prompts/DNABERT.json",
        type=str,
        help="The path to the head formatting for prompting the new explanations",
    )
    parser.add_argument(
        "--full_path",
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/",
        type=str,
        help="The path to the directory",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT",
        type=str,
        help="The model being explained",
    )
    args = parser.parse_args()

    researcher_path = args.researcher_path
    explanation_path = args.explanation_path
    full_path = args.full_path
    model_name = args.model_name
    
    with open(researcher_path, 'r') as f:
        researcher_explanations = json.load(f)

    with open(explanation_path, 'r') as f:
        explanation_prompts = json.load(f)

    prompt_introduction = "We're studying attention head activity in a transformer model for genomics. Each attention head in each layer looks for some particular thing in a DNA sequence. Look at the features that have the absolute value largest coefficients to give the head a name, and also summarize in a SINGLE sentence what the attention head is looking for. Don’t list examples of features, don't summarize any head activity but the last. If there are no features, simply say 'The head is unexplained'. The features format is feature<tab>coefficient. Higher positive values mean a greater positive association with a feature. Higher negative values indicate a greater negative association with a feature. The greater the coefficient, the stronger the match."
    
    if model_name == 'scgpt' or model_name == 'scgpt_pretrained':
        prompt_introduction = "We're studying attention head activity in a transformer model for genomics. Each attention head in each layer looks for some particular thing in a tokenized scRNAseq sequence. Look at the features that have the absolute value largest coefficients give the head a name, and also summarize in a SINGLE sentence what the attention head is looking for. Don’t list examples of features, don't summarize any head activity but the last. If there are no features, simply say 'The head is unexplained'. The features format is feature<tab>coefficient. Higher positive values mean a greater positive association with a feature. Higher negative values indicate a greater negative association with a feature. The greater the coefficient, the stronger the match."

    
    explanations = {}  # dictionary to hold all explanations
    # generate the prompt for each head in explanation_prompts, using few-shot examples
    for head_name, head_data in explanation_prompts.items():
        prompt = prompt_introduction
        example_count = 1

        if head_name in researcher_explanations:
            explanations[head_name] = f'{researcher_explanations[head_name]["given_name"]} : {researcher_explanations[head_name]["explanation"]}'
        else:
            # filter out the current head in case and prepare for random selection
            filtered_heads = {k: v for k, v in researcher_explanations.items() if k != head_name}
            selected_examples = random.sample(list(filtered_heads.items()), min(3, len(filtered_heads)))

            # add few-shot examples from researcher explanations
            for example_head_name, example_head_data in selected_examples:
                prompt += f"\nHead {example_count}\n" + format_coefficients(example_head_data)
                prompt += f"\n{example_head_data['explanation']}\n"
                example_count += 1

            prompt += f"\nHead 4\n" + format_coefficients(head_data) + "This head...\n" + "Please respond with head name and one sentence explanation of the head activity in the format described above.\n"
            print("PROMPT:", prompt)
            #get the explanation from GPT-4s
            explanation = prompt_gpt(prompt)  
            #save
            explanations[head_name] = explanation
            explanation_prefixed = f"{explanation.strip()}"
            explanations[head_name] = explanation_prefixed

    # save to file
    os.makedirs(f'{full_path}/preprocessing/data/explanations/', exist_ok=True)
    with open(f'{full_path}/preprocessing/data/explanations/{model_name}.json', 'w') as file:
        json.dump(explanations, file, indent=4)

    print(f'Explanations saved to {full_path}/preprocessing/data/explanations/{model_name}.json')

if __name__ == "__main__":
    main()