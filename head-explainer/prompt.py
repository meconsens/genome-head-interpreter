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

# Load .env variables
load_dotenv()

# API key from .env
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def get_task_prompt(model_name, task=None):
    """Generate task-specific prompt introduction based on model and task."""
    
    # DNA-based models
    if model_name in ['DNABERT', 'NT']:
        if task == 'enhancers':
            return ("**PROMPT:** Analyze this attention head from a DNA transformer model finetuned to predict enhancer vs non-enhancer sequences.\n"
                   "**Format:**\n"
                   "* Feature name: [Feature coefficient for enhancer, non-enhancer]\n"
                   "* Higher absolute values = stronger association\n"
                   "* Positive = positive association, Negative = negative association\n"
                   "**Instructions:**\n"
                   "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
                   "2. Note how feature importance differs between enhancer and non-enhancer sequences (e.g. if |enhancer coefficient| > |non-enhancer coefficient|, that feature is more relevant in enhancer sequences).\n"
                   "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
                   "**Output ONLY the following format:**\n\n"
                   "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific features and how they differ between enhancer and non-enhancer sequences.]`\n"
                   "**Failsafes:**\n"
                   "* If the head does not meaningfully distinguish enhancer from non-enhancer, say so directly.\n"
                   "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
                   "* Avoid vague phrases like \"associated with\" or \"element sensitivity.\" Be direct and specific.\n"
                   "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")            
        elif task == 'TATA':
            return ("**PROMPT:** Analyze this attention head from a DNA transformer model finetuned to predict TATA-promoter vs non-promoter sequences.\n"
                   "**Format:**\n"
                   "* Feature name: [Feature coefficient for TATA-promoter, non-promoter]\n"
                   "* Higher absolute values = stronger association\n"
                   "* Positive = positive association, Negative = negative association\n"
                   "**Instructions:**\n"
                   "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
                   "2. Note how feature importance differs between TATA-promoter and non-promoter sequences (e.g. if |TATA-promoter coefficient| > |non-promoter coefficient|, that feature is more relevant in TATA-promoter sequences).\n"
                   "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
                   "**Output ONLY the following format:**\n\n"
                   "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific features and how they differ between TATA-promoter and non-promoter sequences.]`\n"
                   "**Failsafes:**\n"
                   "* If the head does not meaningfully distinguish TATA-promoter from non-promoter, say so directly.\n"
                   "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
                   "* Avoid vague phrases like \"associated with\" or \"element sensitivity.\" Be direct and specific.\n"
                   "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")
        else:
            return ("**PROMPT:** Analyze this attention head from a DNA transformer model.\n"
                   "**Format:**\n"
                   "* Feature name: [Feature coefficient values]\n"
                   "* Higher absolute values = stronger association\n"
                   "* Positive = positive association, Negative = negative association\n"
                   "**Instructions:**\n"
                   "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
                   "2. Note how feature importance differs between sample types (e.g. if |coefficient for type A| > |coefficient for type B|, that feature is more relevant in type A).\n"
                   "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
                   "**Output ONLY the following format:**\n\n"
                   "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific features and how they differ between sample types.]`\n"
                   "**Failsafes:**\n"
                   "* If the head does not meaningfully distinguish between sample types, say so directly.\n"
                   "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
                   "* Avoid vague phrases like \"associated with\" or \"element sensitivity.\" Be direct and specific.\n"
                   "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")
    # scGPT models
    elif model_name in ['scgpt', 'scgpt_pretrained']:
        if task == 'ms':
            return ("**PROMPT:** Analyze this attention head from a single-cell transformer model trained to classify central nervous system cell types using data from both healthy and MS samples.\n"
                   "**Format:**\n"
                   "* Feature name: [Feature coefficient values across different cell types]\n"
                   "* Higher absolute values = stronger association\n"
                   "* Positive = positive association, Negative = negative association\n"
                   "**Instructions:**\n"
                   "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
                   "2. Note how feature importance differs between central nervous system cell types (e.g. if |coefficient for cell type A| > |coefficient for cell type B|, that feature is more relevant in cell type A).\n"
                   "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
                   "**Output ONLY the following format:**\n\n"
                   "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific gene expression patterns and which cell types are most distinctly characterized.]`\n"
                   "**Failsafes:**\n"
                   "* If the head does not meaningfully distinguish between cell types, say so directly.\n"
                   "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
                   "* Avoid vague phrases like \"associated with\" or \"sensitivity.\" Be direct and specific.\n"
                   "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")
        elif task == 'pancreas':
            return ("**PROMPT:** Analyze this attention head from a single-cell transformer model finetuned to predict pancreatic cell types.\n"
                   "**Format:**\n"
                   "* Feature name: [Feature coefficient values across different cell types]\n"
                   "* Higher absolute values = stronger association\n"
                   "* Positive = positive association, Negative = negative association\n"
                   "**Instructions:**\n"
                   "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
                   "2. Note how feature importance differs between pancreatic cell types (e.g. if |coefficient for beta cells| > |coefficient for alpha cells|, that feature is more relevant in beta cells).\n"
                   "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
                   "**Output ONLY the following format:**\n\n"
                   "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific gene expression patterns and which pancreatic cell types are most distinctly characterized.]`\n"
                   "**Failsafes:**\n"
                   "* If the head does not meaningfully distinguish between pancreatic cell types, say so directly.\n"
                   "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
                   "* Avoid vague phrases like \"associated with\" or \"sensitivity.\" Be direct and specific.\n"
                   "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")
        else:
            return ("**PROMPT:** Analyze this attention head from a single-cell transformer model.\n"
                   "**Format:**\n"
                   "* Feature name: [Feature coefficient values]\n"
                   "* Higher absolute values = stronger association\n"
                   "* Positive = positive association, Negative = negative association\n"
                   "**Instructions:**\n"
                   "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
                   "2. Note how feature importance differs between cell types (e.g. if |coefficient for cell type A| > |coefficient for cell type B|, that feature is more relevant in cell type A).\n"
                   "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
                   "**Output ONLY the following format:**\n\n"
                   "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific gene expression patterns and how they differ between cell types.]`\n"
                   "**Failsafes:**\n"
                   "* If the head does not meaningfully distinguish between cell types, say so directly.\n"
                   "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
                   "* Avoid vague phrases like \"associated with\" or \"sensitivity.\" Be direct and specific.\n"
                   "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")
        # Default prompt
    else:
        return ("**PROMPT:** Analyze this attention head from a genomic transformer model finetuned for classification.\n"
               "**Format:**\n"
               "* Feature name: [Feature coefficient values]\n"
               "* Higher absolute values = stronger association\n"
               "* Positive = positive association, Negative = negative association\n"
               "**Instructions:**\n"
               "1. Identify the most informative features (those with the largest coefficient magnitudes).\n"
               "2. Note how feature importance differs between sample types (e.g. if |coefficient for type A| > |coefficient for type B|, that feature is more relevant in type A).\n"
               "3. Apply your biological knowledge to provide a **mechanistically meaningful and biologically accurate** explanation.\n"
               "**Output ONLY the following format:**\n\n"
               "`Head Name: [3–5 word mechanistically descriptive name] Explanation: [1–2 plain, specific sentences describing what this head captures biologically. Refer to specific genomic features and how they differ between sample types.]`\n"
               "**Failsafes:**\n"
               "* If the head does not meaningfully distinguish between sample types, say so directly.\n"
               "* If the observed associations contradict established biology, do not try to rationalize or justify them—simply state that the pattern is unclear or inconsistent.\n"
               "* Avoid vague phrases like \"associated with\" or \"element sensitivity.\" Be direct and specific.\n"
               "* Do **not** include any bullet points, summaries, or explanations outside the requested format.")

def format_vectorized_features(head_data):
    """
    Format features as vectors showing coefficients across different contexts
    """
    feature_map = {}
    sentence_types = []
    
    # Identify all sentence types
    for key in head_data:
        if key.endswith('_sentences'):
            # Extract the sentence type (before '_sentences')
            sentence_type = key.replace('_sentences', '')
            sentence_types.append(sentence_type)
    
    # If no specific types found, just use the general sentences
    if not sentence_types and 'sentences' in head_data:
        sentence_types = ['general']
    elif not sentence_types:
        return "No feature data available."
    
    # Initialize with general sentences if available
    if 'sentences' in head_data:
        for sentence in head_data['sentences']:
            for item in sentence:
                if isinstance(item, list) and len(item) == 2:
                    feature_name, value = item
                    feature_map[feature_name] = {'general': value}
    
    # Add type-specific values
    for sentence_type in sentence_types:
        key = f"{sentence_type}_sentences" if sentence_type != 'general' else 'sentences'
        if key in head_data:
            for sentence in head_data[key]:
                for item in sentence:
                    if isinstance(item, list) and len(item) == 2:
                        feature_name, value = item
                        if feature_name not in feature_map:
                            feature_map[feature_name] = {}
                        feature_map[feature_name][sentence_type] = value
    
    # Format the output
    formatted = ""
    for feature, values in feature_map.items():
        # Create vector of values in consistent order
        vector = []
        for sentence_type in sentence_types:
            if sentence_type in values:
                vector.append(str(values[sentence_type]))
            else:
                vector.append("0")  # Default to 0 if not found
        
        formatted += f"{feature}: [{', '.join(vector)}]\n"
    
    return formatted, sentence_types

def prompt_gpt(prompt):
    """Send prompt to OpenAI API and get response"""
    client = OpenAI()
    results = []

    stream = client.chat.completions.create(
        #model="gpt-4-0125-preview",
        model="gpt-4.1-mini-2025-04-14",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            results.append(chunk.choices[0].delta.content)

    # join the results into a single string
    return ''.join(results)

def determine_task(model_name, explanation_path):
    """Determine the task based on model name and data in explanation file"""
    if model_name not in ['DNABERT', 'NT', 'scgpt', 'scgpt_pretrained']:
        return None
    
    # Load explanation data to check for task-specific keys
    with open(explanation_path, 'r') as f:
        explanation_data = json.load(f)
    
    # Get first head data to check structure
    first_head = next(iter(explanation_data.values()))
    
    # DNABERT/NT tasks
    if model_name in ['DNABERT', 'NT']:
        if 'enhancer_sentences' in first_head:
            return 'enhancer'
        elif 'tata_sentences' in first_head or 'TATA_sentences' in first_head:
            return 'tata'
    
    # scGPT tasks
    elif model_name in ['scgpt', 'scgpt_pretrained']:
        # Check for MS-related keys
        ms_indicators = ['SV2C-expressing interneuron_sentences', 'MS_sentences']
        for key in ms_indicators:
            if any(key in head_data for head_data in explanation_data.values()):
                return 'ms'
        
        # Check for pancreas-related keys
        pancreas_indicators = ['alpha_sentences', 'beta_sentences', 'delta_sentences']
        for key in pancreas_indicators:
            if any(key in head_data for head_data in explanation_data.values()):
                return 'pancreas'
    
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--explanation_path",
        default="/path/to/explanations.json",
        type=str,
        help="The path to the head formatting for prompting the new explanations",
    )
    parser.add_argument(
        "--full_path",
        default="/path/to/output/directory",
        type=str,
        help="The path to the directory for outputs",
    )
    parser.add_argument(
        "--model_name",
        default="DNABERT",
        type=str,
        help="The model being explained (DNABERT, NT, scgpt, scgpt_pretrained)",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        help="Optional: Specify task (enhancer, tata, ms, pancreas)",
    )
    args = parser.parse_args()
    
    explanation_path = args.explanation_path
    full_path = args.full_path
    model_name = args.model_name
    
    # If task not specified, try to determine it
    task = args.task if args.task else determine_task(model_name, explanation_path)
    
    with open(explanation_path, 'r') as f:
        explanation_prompts = json.load(f)
    
    # Get the task-specific prompt introduction
    prompt_introduction = get_task_prompt(model_name, task)
    
    explanations = {}  # dictionary to hold all explanations
    
    # Process each head
    for head_name, head_data in explanation_prompts.items():
        # Format features as vectors with coefficients across contexts
        formatted_features, sentence_types = format_vectorized_features(head_data)
        
        # Build the full prompt
        prompt = prompt_introduction
        
        # Add sentence type labels if needed
        if len(sentence_types) > 0:
            sentence_type_legend = f"\nFeature vectors show values for: [{', '.join(sentence_types)}]\n\n"
            prompt += sentence_type_legend
            
        prompt += f"Head data:\n```\n{formatted_features}\n```\n"
        
        print(f"Processing head: {head_name}")
        print("PROMPT:", prompt)
        
        # Get explanation from GPT-4
        explanation = prompt_gpt(prompt)
        explanations[head_name] = explanation.strip()
        print("EXPLANATION:", explanations[head_name])
        return
    
    # Save to file
    os.makedirs(f'{full_path}/explanations/', exist_ok=True)
    with open(f'{full_path}/explanations/{model_name}_vectorized.json', 'w') as file:
        json.dump(explanations, file, indent=4)
    
    print(f'Explanations saved to {full_path}/explanations/{model_name}_vectorized.json')

if __name__ == "__main__":
    main()