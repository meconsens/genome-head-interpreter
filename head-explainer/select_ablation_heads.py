import pandas as pd
import numpy as np
import argparse
import os
import re
import random
from collections import defaultdict

def load_zscores(full_path, model_name, subtype, prefix="centered_"):
    """
    Load z-scores for a specific model subtype (global and label-specific).
    """
    # Load global z-scores
    z_score_path = f"{full_path}/data/z_scores/{model_name}/{subtype}_{prefix}z_scores.csv"
    if not os.path.exists(z_score_path):
        # Try without prefix
        z_score_path = f"{full_path}/data/z_scores/{model_name}/{subtype}_z_scores.csv"
        if not os.path.exists(z_score_path):
            print(f"No z-score file found for {subtype}. Skipping.")
            return None, {}
    
    global_zscores = pd.read_csv(z_score_path, index_col=0)
    
    # Load label-specific z-scores
    label_zscores = {}
    label_specific_dir = f"{full_path}/data/z_scores/{model_name}/{subtype}/label_specific/"
    
    if os.path.exists(label_specific_dir):
        # Check for files with prefix first
        label_files = [f for f in os.listdir(label_specific_dir) 
                      if f.endswith(f"{prefix}z_scores.csv")]
        
        # If no files with prefix, try without prefix
        if not label_files:
            label_files = [f for f in os.listdir(label_specific_dir) 
                          if f.endswith("_z_scores.csv")]
        
        for label_file in label_files:
            # Extract label from filename
            if prefix in label_file:
                label = label_file.replace(f"_{prefix}z_scores.csv", "")
            else:
                label = label_file.replace("_z_scores.csv", "")
            
            label_df = pd.read_csv(os.path.join(label_specific_dir, label_file), index_col=0)
            label_zscores[label] = label_df
    
    return global_zscores, label_zscores

def extract_layer_head(head_name):
    """
    Extract layer and head numbers from head name.
    Example: 'L0.H6' -> (0, 6)
    """
    # Look for patterns like L0.H6 or l0-h6 or layer0.head6 or layer0_head6
    pattern = r'[lL](?:ayer)?[-_.\s]*(\d+)[-_.\s]*[hH](?:ead)?[-_.\s]*(\d+)'
    match = re.search(pattern, head_name)
    
    if match:
        layer_num = int(match.group(1))
        head_num = int(match.group(2))
        return layer_num, head_num
    else:
        # Alternative pattern if head name is simply a number (assume global index)
        if head_name.isdigit():
            head_idx = int(head_name)
            # Assuming 12 heads per layer, convert global index to layer/head
            layer_num = head_idx // 12
            head_num = head_idx % 12
            return layer_num, head_num
        
        # If no match found, return None
        print(f"Warning: Could not extract layer and head from '{head_name}'")
        return None, None

def calculate_head_importance_with_features(zscore_df):
    """
    Calculate importance score for each head based on feature associations,
    also tracking which feature has the strongest association.
    """
    # For each head, find the feature with max absolute z-score
    strongest_features = {}
    strongest_zscores = {}
    
    for head_name in zscore_df.index:
        head_scores = zscore_df.loc[head_name].abs()
        max_feature = head_scores.idxmax()
        max_zscore = zscore_df.loc[head_name, max_feature]
        strongest_features[head_name] = max_feature
        strongest_zscores[head_name] = max_zscore
    
    # Calculate importance metrics
    head_importance = pd.DataFrame({
        'head_name': zscore_df.index,
        'mean_abs_zscore': zscore_df.abs().mean(axis=1),
        'max_abs_zscore': zscore_df.abs().max(axis=1),
        'num_significant': (zscore_df.abs() > 3).sum(axis=1),  # Count z-scores > 3
        'strongest_feature': [strongest_features[head] for head in zscore_df.index],
        'strongest_zscore': [strongest_zscores[head] for head in zscore_df.index]
    })
    
    # Extract layer and head numbers
    layer_head_info = [extract_layer_head(head) for head in head_importance['head_name']]
    head_importance['layer'] = [info[0] if info[0] is not None else -1 for info in layer_head_info]
    head_importance['head'] = [info[1] if info[1] is not None else -1 for info in layer_head_info]
    
    # Filter out heads where layer/head extraction failed
    head_importance = head_importance[head_importance['layer'] >= 0]
    
    # Sort by importance score
    head_importance = head_importance.sort_values('max_abs_zscore', ascending=False)
    
    return head_importance

def select_feature_specific_paired_heads(zscore_df, importance_df, feature_name, num_top_heads=10, sign_filter=None):
    """
    Select top N heads with strongest z-score associations to a specific feature,
    and for each, find a matching head from the same layer with weakest association
    to that same feature.
    
    Args:
        zscore_df: Original DataFrame with z-scores (heads as index, features as columns)
        importance_df: DataFrame with head importance scores and strongest features
        feature_name: Name of the feature to analyze
        num_top_heads: Number of top heads to select
        sign_filter: Filter for "positive", "negative", or None (absolute) associations
        
    Returns:
        important_heads: List of important head dictionaries with feature info
        unimportant_heads: List of matched unimportant head dictionaries from same layers
    """
    # Check if feature exists
    if feature_name not in zscore_df.columns:
        available_features = list(zscore_df.columns)
        raise ValueError(f"Feature '{feature_name}' not found in z-score data. Available features: {available_features[:10]} (and {len(available_features)-10} more)")
    
    # Create a DataFrame with layer, head, and z-score for the specified feature
    feature_scores = pd.DataFrame({
        'head_name': zscore_df.index,
        'feature_zscore': zscore_df[feature_name],
        'abs_feature_zscore': zscore_df[feature_name].abs()
    })
    
    # Add layer and head info
    layer_head_info = [extract_layer_head(head) for head in feature_scores['head_name']]
    feature_scores['layer'] = [info[0] if info[0] is not None else -1 for info in layer_head_info]
    feature_scores['head'] = [info[1] if info[1] is not None else -1 for info in layer_head_info]
    
    # Filter out heads where layer/head extraction failed
    feature_scores = feature_scores[feature_scores['layer'] >= 0]
    
    # Merge with importance metrics for additional context
    feature_scores = pd.merge(
        feature_scores,
        importance_df[['layer', 'head', 'mean_abs_zscore', 'max_abs_zscore']],
        on=['layer', 'head'],
        how='left'
    )
    
    # Apply sign filtering if specified
    if sign_filter == "positive":
        feature_scores = feature_scores[feature_scores['feature_zscore'] > 0]
        # Now sort by the actual z-score (not absolute value) for positive associations
        sort_column = 'feature_zscore'
        sort_ascending = False  # Highest positive scores first
    elif sign_filter == "negative":
        feature_scores = feature_scores[feature_scores['feature_zscore'] < 0]
        # For negative associations, sort by the actual z-score (most negative first)
        sort_column = 'feature_zscore'
        sort_ascending = True  # Lowest (most negative) scores first
    else:
        # Default: sort by absolute z-score for this feature (strongest association)
        sort_column = 'abs_feature_zscore'
        sort_ascending = False  # Highest absolute scores first
    
    # Sort based on the selected criteria
    feature_scores_sorted = feature_scores.sort_values(sort_column, ascending=sort_ascending)
    
    # Select top N heads for this feature
    top_heads = feature_scores_sorted.head(num_top_heads)
    
    important_heads = []
    unimportant_heads = []
    
    # For each top head, find the head from the same layer with weakest association to this feature
    for _, top_head in top_heads.iterrows():
        # Create the important head entry
        important_heads.append({
            "layer": int(top_head['layer']),
            "head": int(top_head['head']),
            "name": f"Layer{int(top_head['layer'])}-Head{int(top_head['head'])}",
            "feature": feature_name,
            "feature_zscore": float(top_head['feature_zscore']),
            "abs_feature_zscore": float(top_head['abs_feature_zscore']),
            "mean_abs_zscore": float(top_head['mean_abs_zscore'])
        })
        
        # Find heads from the same layer
        same_layer_heads = feature_scores[feature_scores['layer'] == top_head['layer']]
        
        # Determine how to find the contrasting head based on sign_filter
        if sign_filter == "positive":
            # For positive associations, the weakest would be the most negative or closest to zero
            same_layer_sorted = same_layer_heads.sort_values('feature_zscore', ascending=True)
        elif sign_filter == "negative":
            # For negative associations, the weakest would be the most positive or closest to zero
            same_layer_sorted = same_layer_heads.sort_values('feature_zscore', ascending=False)
        else:
            # For absolute associations, the weakest would be closest to zero
            same_layer_sorted = same_layer_heads.sort_values('abs_feature_zscore', ascending=True)
        
        # Find a different head in the same layer with the weakest association to this feature
        found_weak_head = False
        for _, weak_head in same_layer_sorted.iterrows():
            if weak_head['head'] != top_head['head']:
                unimportant_heads.append({
                    "layer": int(weak_head['layer']),
                    "head": int(weak_head['head']),
                    "name": f"Layer{int(weak_head['layer'])}-Head{int(weak_head['head'])}",
                    "feature": feature_name,
                    "feature_zscore": float(weak_head['feature_zscore']),
                    "abs_feature_zscore": float(weak_head['abs_feature_zscore']),
                    "mean_abs_zscore": float(weak_head['mean_abs_zscore'])
                })
                found_weak_head = True
                break
        
        # If we couldn't find a different head in the same layer
        if not found_weak_head:
            # Get the globally least associated head for this feature
            if sign_filter == "positive":
                least_important = feature_scores.sort_values('feature_zscore', ascending=True).iloc[0]
            elif sign_filter == "negative":
                least_important = feature_scores.sort_values('feature_zscore', ascending=False).iloc[0]
            else:
                least_important = feature_scores.sort_values('abs_feature_zscore', ascending=True).iloc[0]
            
            unimportant_heads.append({
                "layer": int(least_important['layer']),
                "head": int(least_important['head']),
                "name": f"Layer{int(least_important['layer'])}-Head{int(least_important['head'])}",
                "feature": feature_name,
                "feature_zscore": float(least_important['feature_zscore']),
                "abs_feature_zscore": float(least_important['abs_feature_zscore']),
                "mean_abs_zscore": float(least_important['mean_abs_zscore']),
                "note": "No alternative head in same layer, using globally least associated head"
            })
    
    return important_heads, unimportant_heads

def main():
    parser = argparse.ArgumentParser(description='Select heads for ablation testing based on feature associations')
    parser.add_argument("--full_path", default="/home/mica/genome-head-interpreter/preprocessing", type=str)
    parser.add_argument("--model_name", default="NT_fake_TATA", type=str,
                        help="Model name to analyze")
    parser.add_argument("--subtype", default="finetuned", type=str,
                        help="Model subtype to analyze (default: finetuned)")
    parser.add_argument("--prefix", default="centered_", type=str,
                        help="Prefix for z-score files (default: 'centered_')")
    parser.add_argument("--num_top_heads", default=232, type=int,
                        help="Number of top heads to select (default: 96 for scgpt, 72 for DNABERT, 232 for NT)")
    parser.add_argument("--feature", default='GC',type=str,
                        help="Specific feature to analyze for head associations")
    parser.add_argument("--sign_filter", type=str, choices=["positive", "negative"],
                        help="Filter for only positive or negative associations")
    parser.add_argument("--list_features", action="store_true",
                        help="List available features and exit")
    parser.add_argument("--output_file", default="/home/mica/genome-head-interpreter/preprocessing/ablation_heads/", type=str,
                        help="Output Python file with head lists")
    args = parser.parse_args()
    
    # Load z-scores
    global_zscores, label_zscores = load_zscores(args.full_path, args.model_name, args.subtype, args.prefix)
    
    if global_zscores is None:
        print(f"No z-scores found for {args.model_name} {args.subtype}. Exiting.")
        return
    
    # List available features if requested
    if args.list_features:
        print("\nAvailable features:")
        for feature in global_zscores.columns:
            print(f"  {feature}")
        return
    
    # Calculate head importance for global context with strongest feature info
    print("Calculating global head importance with strongest feature information...")
    global_importance = calculate_head_importance_with_features(global_zscores)
    
    # Prepare sign filter message
    sign_msg = ""
    if args.sign_filter:
        sign_msg = f" (filtering for {args.sign_filter} associations)"
    
    # If a specific feature is provided, use feature-specific pairing
    if args.feature:
        print(f"Analyzing associations with feature: {args.feature}{sign_msg}")
        try:
            important_heads, unimportant_heads = select_feature_specific_paired_heads(
                global_zscores, global_importance, args.feature, args.num_top_heads, args.sign_filter)
                
            # Add sign filter to output filename if provided
            sign_suffix = f"_{args.sign_filter}" if args.sign_filter else ""
            output_file = f"{args.output_file}{args.model_name}_{args.subtype}_{args.feature}{sign_suffix}_paired_ablation_heads.py"
        except ValueError as e:
            print(f"Error: {e}")
            return
    else:
        # Fall back to strongest feature across all features if no specific feature provided
        print("No specific feature provided. Analyzing strongest associations across all features.")
        print("To analyze a specific feature, use --feature FEATURE_NAME")
        print("To see available features, use --list_features")
        
        important_heads = []
        unimportant_heads = []
        
        # Get the top N features with the strongest associations across all heads
        all_features = pd.DataFrame({
            'feature': global_zscores.columns,
            'max_abs_zscore': global_zscores.abs().max()
        }).sort_values('max_abs_zscore', ascending=False)
        
        top_features = all_features.head(min(5, len(all_features)))
        print("\nTop features with strongest associations:")
        for _, row in top_features.iterrows():
            print(f"  {row['feature']} (max abs z-score: {row['max_abs_zscore']:.2f})")
        
        # Get the most strongly associated heads for the top feature
        top_feature = top_features.iloc[0]['feature']
        print(f"\nUsing top feature: {top_feature}{sign_msg}")
        
        important_heads, unimportant_heads = select_feature_specific_paired_heads(
            global_zscores, global_importance, top_feature, args.num_top_heads, args.sign_filter)
        
        # Add sign filter to output filename if provided
        sign_suffix = f"_{args.sign_filter}" if args.sign_filter else ""
        output_file = f"{args.output_file}{args.model_name}_{args.subtype}{sign_suffix}_paired_ablation_heads.py"
    
    # Make sure dir exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(f"Writing results to {output_file}...")
    
    with open(output_file, 'w') as f:
        f.write("# Automatically generated head selection for ablation testing\n")
        f.write(f"# Important heads with {args.sign_filter if args.sign_filter else 'strong'} associations to feature '{args.feature if args.feature else top_feature}'\n")
        f.write("# and paired unimportant heads with weak associations to the same feature\n\n")
        
        # Write important heads
        f.write("important_heads = [\n")
        for head in important_heads:
            f.write(f"    {repr(head)},\n")
        f.write("]\n\n")
        
        # Write unimportant heads
        f.write("unimportant_heads = [\n")
        for head in unimportant_heads:
            f.write(f"    {repr(head)},\n")
        f.write("]\n")
    
    # Print summary
    sign_description = f"{args.sign_filter if args.sign_filter else 'strong'}"
    print("\nHead selection complete!")
    print(f"Selected {len(important_heads)} important heads with {sign_description} associations and {len(unimportant_heads)} unimportant heads")
    print(f"Results written to {output_file}")
    
    # Print the important heads with their feature associations
    feature_name = args.feature if args.feature else top_feature
    print(f"\nImportant heads with {sign_description} associations to '{feature_name}':")
    for head in important_heads:
        print(f"  Layer {head['layer']}, Head {head['head']} - Z-score for '{feature_name}': {head['feature_zscore']:.2f}")
    
    print(f"\nUnimportant paired heads with weak associations to '{feature_name}':")
    for head in unimportant_heads:
        print(f"  Layer {head['layer']}, Head {head['head']} - Z-score for '{feature_name}': {head['feature_zscore']:.2f}")

if __name__ == "__main__":
    main()