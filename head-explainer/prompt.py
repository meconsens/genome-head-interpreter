import pandas as pd
import csv
import numpy as np
import argparse
import json
from openai import OpenAI
import dotenv
from dotenv import load_dotenv
import os

#load .env variables
load_dotenv()

# API key from .env
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# coefficients into feature<tab>coef
def format_coefficients(head_data):
    formatted = "Coefficients:\n<start>\n"
    for sentence in head_data["sentences"]:
        for item in sentence:
            if isinstance(item, list):
                formatted += f"{item[0]}\t{item[1]}\n"
            else:
                formatted += f"{item}\t0\n"
    formatted += "<end>\n"
    return formatted

def prompt_gpt(prompt):
    client = OpenAI()
    results = []

    stream = client.chat.completions.create(
        model="gpt-4",
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
        default="/scratch/ssd004/scratch/mconsens/genome-head-interpreter/head-explainer/",
        type=str,
        help="The full path to the format_head_scores.py file",
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

    prompt_introduction = """We're studying attention head activity in a transformer model for genomics. Each attention head in each layer looks for some particular thing in a DNA sequence. Look at the features that have the biggest coefficients and summarize in a single sentence what the attention head is looking for. Don’t list examples of features. The features format is feature\tcoefficient. Higher positive values mean a greater positive association with a feature. Higher negative values indicate a greater negative association with a feature. The greater the coefficient, the stronger the match."""

    explanations = {}  # dictionary to hold all explanations
    # generate the prompt for each head in explanation_prompts, using few-shot examples
    for head_name, head_data in explanation_prompts.items():
        prompt = prompt_introduction
        example_count = 1

        # add few-shot examples from researcher explanations
        for example_head_name, example_head_data in researcher_explanations.items():
            if example_count > 3 or example_head_name == head_name:  # limit to 3 examples, skip if it's the current head
                break
            prompt += f"\nHead {example_count}\n" + format_coefficients(example_head_data)
            prompt += f"\nExplanation of Head {example_count} behavior: {example_head_data['explanation']}\n"
            example_count += 1

        # add current head as the one to explain (if it doesn't have a researcher explanation)
        if head_name not in researcher_explanations:
            prompt += f"\nHead 4\n" + format_coefficients(head_data)
            prompt += "\nExplanation of Head 4 behavior: the main thing this head does is find...."

        #get the explanation from GPT-4
        explanation = prompt_gpt(prompt)  

        #save
        explanations[head_name] = explanation
        explanation_prefixed = f"The main thing this head does is find {explanation.strip()}"
        explanations[head_name] = explanation_prefixed

    # save to file
    with open(f'/scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/explanations/{model_name}.json', 'w') as file:
        json.dump(explanations, file, indent=4)

    print(f'Explanations saved to /scratch/ssd004/scratch/mconsens/genome-head-interpreter/preprocessing/data/explanations/{model_name}.json')

if __name__ == "__main__":
    main()