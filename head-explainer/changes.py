import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import importlib.util
import sys
import re

def load_feature_colors(full_path):
    """
    Load feature color mapping from the feature_color_map.py file.
    
    Args:
        full_path: Base path where feature_color_map.py is located
        
    Returns:
        feature_colors: Dictionary mapping features to colors
        model_legend_groups: Dictionary with model type legend groups
        get_feature_color: Function to get color for a feature
    """
    #color_map_path = os.path.join(full_path, "feature_color_map.py")
    color_map_path = '/home/mica/genome-head-interpreter/head-explainer/feature_color_map.py'
    
    if not os.path.exists(color_map_path):
        print(f"Warning: No feature color map found at {color_map_path}. Using default colors.")
        # Return empty dictionaries and a simple default function
        return {}, {}, lambda feature, model_type: "#377EB8"  # Default blue color
    
    # Load the module
    spec = importlib.util.spec_from_file_location("feature_color_map", color_map_path)
    color_map = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(color_map)
    
    return color_map.feature_colors, color_map.model_legend_groups, color_map.get_feature_color

def get_model_type(model_name):
    """Determine model type from name for color grouping"""
    if model_name.lower().startswith("dnabert"):
        return "DNABERT"
    elif model_name.lower().startswith("nt"):
        return "NT"
    elif model_name.lower().startswith("scgpt"):
        return "scgpt"
    else:
        return "other"

def load_zscores(full_path, model_name, subtype, prefix="centered_"):
    """
    Load z-scores for a specific model subtype (global and label-specific).
    
    Args:
        full_path: Base path for data files
        model_name: Name of the model
        subtype: Model subtype (e.g., random_init, pretrained, finetuned)
        prefix: Prefix for z-score files (default: "centered_")
        
    Returns:
        global_zscores: DataFrame with global z-scores
        label_zscores: Dictionary mapping labels to DataFrames with label-specific z-scores
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

def plot_zscores_comparison(first_zscores, second_zscores, output_path, title, 
                            first_name, second_name, model_name,
                            feature_colors, model_legend_groups, get_feature_color,
                            features=None, heads=None, threshold=3.0):
    """
    Create a scatter plot comparing z-scores between two model subtypes with custom colors.
    Uses compact legend that only shows category groups.
    
    Args:
        first_zscores: DataFrame with z-scores from first model subtype
        second_zscores: DataFrame with z-scores from second model subtype
        output_path: Path to save the output plot
        title: Title for the plot
        first_name: Name of the first model subtype for labeling
        second_name: Name of the second model subtype for labeling
        model_name: Name of the model for color mapping
        feature_colors: Dictionary mapping features to colors
        model_legend_groups: Dictionary with model type legend groups
        get_feature_color: Function to get color for a feature
        features: List of features to include (default: all)
        heads: List of heads to include (default: all)
        threshold: Z-score threshold for highlighting points (default: 3.0)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    
    # Find common heads and features
    common_heads = set(first_zscores.index).intersection(set(second_zscores.index))
    common_features = set(first_zscores.columns).intersection(set(second_zscores.columns))
    
    if not common_heads or not common_features:
        print(f"No common heads or features found for {title}. Skipping plot.")
        return
    
    # Filter by specified heads and features if provided
    if heads is not None:
        common_heads = common_heads.intersection(heads)
    if features is not None:
        common_features = common_features.intersection(features)
    
    # Convert sets to lists for pandas indexing
    common_heads_list = list(common_heads)
    common_features_list = list(common_features)
    
    # Filter DataFrames to common heads and features
    first_filtered = first_zscores.loc[common_heads_list, common_features_list]
    second_filtered = second_zscores.loc[common_heads_list, common_features_list]
    
    # Flatten DataFrames to 1D arrays for plotting
    x_values = first_filtered.values.flatten()
    y_values = second_filtered.values.flatten()
    
    # Create feature-head pair labels for annotations
    head_feature_pairs = [(head, feature) for head in common_heads_list for feature in common_features_list]
    
    # Calculate absolute changes
    abs_changes = np.abs(y_values - x_values)
    
    # Determine significant points (above threshold)
    # Calculate standard deviation of changes
    std_change = np.std(abs_changes)
    print(f"Standard deviation of changes: {std_change:.4f}")
    significant_threshold = threshold * std_change
    is_significant = abs_changes > significant_threshold
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # First plot all points as gray
    plt.scatter(x_values, y_values, c='lightgray', alpha=0.5, s=30)
    
    # Get model type for color mapping
    model_type = get_model_type(model_name)
    print(f"Model type identified as: {model_type}")
    
    # Track feature categories used in this plot
    categories_used = set()
    feature_to_category = {}
    
    # Plot significant points with feature-based colors
    if np.any(is_significant):
        for idx, (is_sig, (head, feature)) in enumerate(zip(is_significant, head_feature_pairs)):
            if is_sig:
                try:
                    # Get the color for this feature
                    color = get_feature_color(feature, model_type)
                    
                    # Determine feature category for legend grouping
                    category = determine_feature_category(feature, model_type)
                    categories_used.add(category)
                    feature_to_category[feature] = category
                    
                    plt.scatter(x_values[idx], y_values[idx], 
                               c=[color], 
                               s=50, alpha=0.8)
                    
                    # Annotate points with extreme changes
                    if abs_changes[idx] > 2 * significant_threshold:
                        plt.annotate(f"{head}",
                                   (x_values[idx], y_values[idx]),
                                   fontsize=8, alpha=0.7)
                except Exception as e:
                    print(f"Error plotting point for feature {feature}: {str(e)}")
    
    # Add diagonal line (y=x)
    min_val = min(np.min(x_values), np.min(y_values))
    max_val = max(np.max(x_values), np.max(y_values))
    margin = (max_val - min_val) * 0.1
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 
             'k--', alpha=0.5, label='y=x')
    
    # Add horizontal and vertical lines at y=0 and x=0
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Set equal scaling for x and y axes
    plt.axis('equal')
    
    # Labels and title
    plt.xlabel(f'{first_name} Z-scores', fontsize=14)
    plt.ylabel(f'{second_name} Z-scores', fontsize=14)
    plt.title(title, fontsize=16)
    
    # Create a simplified legend with just category groups
    if categories_used:
        legend_elements = []
        
        # Define default colors in case model_legend_groups is incomplete
        default_category_colors = {
            "PhyloP": "#D62728",                        # red
            "TSS": "#2CA02C",                           # green
            "GC": "#9467BD",                            # purple
            "Repeats": "#7F8C8D",                       # gray
            "Zinc Finger Factors": "#1F77B4",           # blue
            "Helix Factors": "#27AE60",                 # emerald
            "Homeodomain Factors": "#8C564B",           # brown
            "Leucine Zipper Factors": "#1ABC9C",        # teal
            "Receptor-like Factors": "#8E44AD",         # purple
            "Other Transcription Factors": "#F39C12",   # orange
            "Expression Features": "#7570B3",           # purple
            "Cell Types": "#1B9E77",                    # teal
            "Biological Processes (GOBP)": "#66A61E",   # green
            "Cell Components (GOCC)": "#D95F02",        # orange
            "Molecular Functions (GOMF)": "#E41A1C"     # red
        }
        
        # Only show category headers with their representative colors
        print(f"Categories used: {categories_used}")
        for category in sorted(categories_used):
            # Get color from model_legend_groups if available, otherwise use default
            color = None
            
            # For DNABERT/NT models, check nested structure
            if model_type in ["DNABERT", "NT"] and model_type in model_legend_groups:
                # These models have a nested structure in model_legend_groups
                for group_name, group_dict in model_legend_groups[model_type].items():
                    if group_name == category and isinstance(group_dict, dict):
                        # Take first color from the nested dict
                        color = next(iter(group_dict.values()))
                        break
                    elif group_name == category:
                        # Direct string color
                        color = group_dict
                        break
            # For non-nested structure (scGPT)
            elif model_type in model_legend_groups and category in model_legend_groups[model_type]:
                color = model_legend_groups[model_type][category]
            
            # Use default if not found
            if not color and category in default_category_colors:
                color = default_category_colors[category]
            elif not color:
                # Use a hardcoded default color as last resort
                color = "#777777"  # gray
                
            print(f"Using color {color} for category {category}")
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                       markerfacecolor=color, markersize=10,
                                       label=category))
        
        plt.legend(handles=legend_elements, loc="upper right", ncol=1)
    
    # Grid lines
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_path}")
    print(f"Number of significant points: {np.sum(is_significant)} out of {len(is_significant)}")
    
    # Return significant points for further analysis
    significant_indices = np.where(is_significant)[0]
    significant_data = []
    for idx in significant_indices:
        head, feature = head_feature_pairs[idx]
        significant_data.append({
            'head': head,
            'feature': feature,
            'first_subtype': x_values[idx],
            'second_subtype': y_values[idx],
            'abs_change': abs_changes[idx]
        })
    
    return significant_data

def get_model_type(model_name):
    """Determine model type from name for color grouping"""
    if model_name.lower().startswith("dnabert"):
        return "DNABERT"
    elif model_name.lower().startswith("nt"):
        return "NT"
    elif model_name.lower().startswith("scgpt"):
        return "scgpt"
    else:
        return "other"

def determine_feature_category(feature_name, model_type):
    """Determine the category of a feature for legend grouping"""
    if model_type in ["DNABERT", "NT"]:
        if feature_name == "phyloP":
            return "PhyloP"
        elif feature_name == "TSS":
            return "TSS"
        elif feature_name == "GC":
            return "GC"
        elif feature_name.startswith("repeat_"):
            return "Repeats"
        else:
            # For transcription factors, determine the family
            tf_lower = feature_name.lower()
            
            # Zinc finger family
            if "zf" in tf_lower or "zinc" in tf_lower or "c2h2" in tf_lower or "gata" in tf_lower:
                return "Zinc Finger Factors"
                
            # Basic helix factors
            elif "bhlh" in tf_lower or "helix" in tf_lower:
                return "Helix Factors"
                
            # Homeodomain factors
            elif "homeo" in tf_lower or "hox" in tf_lower or "fork" in tf_lower:
                return "Homeodomain Factors"
                
            # Leucine zipper factors
            elif "zip" in tf_lower or "leucine" in tf_lower or "bzip" in tf_lower:
                return "Leucine Zipper Factors"
                
            # Receptor-like factors
            elif "rxr" in tf_lower or "receptor" in tf_lower or "thr" in tf_lower:
                return "Receptor-like Factors"
            
            else:
                return "Other Transcription Factors"
    
    elif model_type == "scgpt":
        if feature_name == "expression":
            return "Expression Features"
        elif feature_name.startswith("GOBP"):
            return "Biological Processes (GOBP)"
        elif feature_name.startswith("GOCC"):
            return "Cell Components (GOCC)"
        elif feature_name.startswith("GOMF"):
            return "Molecular Functions (GOMF)"
        else:
            return "Cell Types"
    else:
        return "Other"


def plot_zscores_heatmap_diff(first_zscores, second_zscores, output_path, title, 
                              first_name, second_name, features=None, heads=None):
    """
    Create a heatmap showing the difference between z-scores of two model subtypes.
    
    Args:
        first_zscores: DataFrame with z-scores from first model subtype
        second_zscores: DataFrame with z-scores from second model subtype
        output_path: Path to save the output plot
        title: Title for the plot
        first_name: Name of the first model subtype for labeling
        second_name: Name of the second model subtype for labeling
        features: List of features to include (default: all)
        heads: List of heads to include (default: all)
    """
    # Find common heads and features
    common_heads = set(first_zscores.index).intersection(set(second_zscores.index))
    common_features = set(first_zscores.columns).intersection(set(second_zscores.columns))
    
    if not common_heads or not common_features:
        print(f"No common heads or features found for {title}. Skipping plot.")
        return
    
    # Filter by specified heads and features if provided
    if heads is not None:
        common_heads = common_heads.intersection(heads)
    if features is not None:
        common_features = common_features.intersection(features)
    
    # Convert sets to lists for pandas indexing
    common_heads_list = list(common_heads)
    common_features_list = list(common_features)
    
    # Filter DataFrames to common heads and features
    first_filtered = first_zscores.loc[common_heads_list, common_features_list]
    second_filtered = second_zscores.loc[common_heads_list, common_features_list]
    
    # Calculate differences (second - first)
    diff_df = second_filtered - first_filtered
    
    # Create the heatmap
    plt.figure(figsize=(max(12, len(common_features) * 0.4), max(10, len(common_heads) * 0.4)))
    
    # Define a diverging colormap (blue-white-red)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    
    # Get the maximum absolute value for symmetric color scaling
    max_abs_val = max(abs(diff_df.values.min()), abs(diff_df.values.max()))
    
    # Create heatmap
    sns.heatmap(diff_df, cmap=cmap, center=0, vmin=-max_abs_val, vmax=max_abs_val,
                annot=False, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Z-score Difference"})
    
    plt.title(f"{title}\n({second_name} - {first_name})", fontsize=16)
    plt.ylabel('Heads', fontsize=14)
    plt.xlabel('Features', fontsize=14)
    
    # Rotate x-axis labels if many features
    if len(common_features) > 10:
        plt.xticks(rotation=90)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved difference heatmap to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare feature-head association z-scores across model training stages')
    parser.add_argument("--data_path", default="/home/mica/genome-head-interpreter/preprocessing/data/coef/", type=str)
    parser.add_argument("--full_path", default="/home/mica/genome-head-interpreter/preprocessing/", type=str)
    parser.add_argument("--model_name", default="DNABERT_TATA", type=str)
    parser.add_argument("--output_dir", default=None, type=str,
                        help="Directory to save the output visualizations")
    parser.add_argument("--prefix", default="centered_", type=str,
                        help="Prefix for z-score files (default: 'centered_')")
    parser.add_argument("--plot_heads", default=None, type=str,
                        help="Comma-separated list of heads to include (default: all)")
    parser.add_argument("--plot_features", default=None, type=str,
                        help="Comma-separated list of features to include (default: all)")
    parser.add_argument("--threshold", default=3.0, type=float,
                        help="Threshold factor (in standard deviations) for highlighting significant changes (default: 3.0)")
    parser.add_argument("--first_subtype", default="pretrained", type=str,
                        help="First model subtype to compare (default: pretrained)")
    parser.add_argument("--second_subtype", default="finetuned", type=str,
                        help="Second model subtype to compare (default: finetuned)")
    parser.add_argument("--allow_random", action="store_true",
                        help="Allow random_init subtype for any model (by default, only allowed for DNABERT)")
    args = parser.parse_args()
    
    # Load feature color mapping
    feature_colors, model_legend_groups, get_feature_color = load_feature_colors(args.full_path)
    
    # Validate subtype combinations
    if (args.first_subtype == "random" or args.second_subtype == "random") and \
       not args.allow_random and not "dnabert" in args.model_name.lower():
        print("Warning: random subtype is only allowed for DNABERT models unless --allow_random is specified.")
        print("If you want to use random for other models, add --allow_random to your command.")
        if not "dnabert" in args.model_name.lower():
            return
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"{args.full_path}/visualizations/{args.model_name}/{args.first_subtype}_vs_{args.second_subtype}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse heads and features if provided
    heads = None
    if args.plot_heads is not None:
        heads = set(args.plot_heads.split(','))
    
    features = None
    if args.plot_features is not None:
        features = set(args.plot_features.split(','))
    
    # Load z-scores for both model subtypes
    first_global, first_label = load_zscores(args.full_path, args.model_name, args.first_subtype, args.prefix)
    second_global, second_label = load_zscores(args.full_path, args.model_name, args.second_subtype, args.prefix)
    
    if first_global is None or second_global is None:
        print(f"Error: Could not load global z-scores for {args.first_subtype} or {args.second_subtype} models.")
        return
    
    # Create formatted names for display
    first_display_name = args.first_subtype.capitalize()
    second_display_name = args.second_subtype.capitalize()
    
    # Plot global z-scores comparison
    print(f"Plotting global z-scores comparison between {args.first_subtype} and {args.second_subtype}...")
    significant_global = plot_zscores_comparison(
        first_global, 
        second_global,
        os.path.join(args.output_dir, "global_zscores_comparison.png"),
        f"{args.model_name}: Global {first_display_name} vs. {second_display_name} Z-scores",
        first_display_name,
        second_display_name,
        args.model_name,
        feature_colors,
        model_legend_groups,
        get_feature_color,
        features,
        heads,
        args.threshold
    )
    
    # Also plot difference heatmap
    plot_zscores_heatmap_diff(
        first_global, 
        second_global,
        os.path.join(args.output_dir, "global_zscores_diff_heatmap.png"),
        f"{args.model_name}: Global Z-score Differences",
        first_display_name,
        second_display_name,
        features,
        heads
    )
    
    # Find common labels between the two subtypes
    common_labels = set(first_label.keys()).intersection(set(second_label.keys()))
    
    # Collect all significant changes
    all_significant = []
    if significant_global:
        for item in significant_global:
            item['context'] = 'global'
            item[args.first_subtype] = item.pop('first_subtype')  # Rename keys to match subtype names
            item[args.second_subtype] = item.pop('second_subtype')
            all_significant.append(item)
    
    # Plot label-specific z-scores comparisons
    for label in common_labels:
        print(f"Plotting z-scores comparison for label '{label}'...")
        significant_label = plot_zscores_comparison(
            first_label[label], 
            second_label[label],
            os.path.join(args.output_dir, f"label_{label}_zscores_comparison.png"),
            f"{args.model_name}: Label '{label}' {first_display_name} vs. {second_display_name} Z-scores",
            first_display_name,
            second_display_name,
            args.model_name,
            feature_colors,
            model_legend_groups,
            get_feature_color,
            features,
            heads,
            args.threshold
        )
        
        # Add to all significant changes
        if significant_label:
            for item in significant_label:
                item['context'] = f'label_{label}'
                item[args.first_subtype] = item.pop('first_subtype')  # Rename keys to match subtype names
                item[args.second_subtype] = item.pop('second_subtype')
                all_significant.append(item)
        
        # Also plot difference heatmap
        plot_zscores_heatmap_diff(
            first_label[label], 
            second_label[label],
            os.path.join(args.output_dir, f"label_{label}_zscores_diff_heatmap.png"),
            f"{args.model_name}: Label '{label}' Z-score Differences",
            first_display_name,
            second_display_name,
            features,
            heads
        )
    
    # Save significant changes to CSV
    if all_significant:
        significant_df = pd.DataFrame(all_significant)
        significant_df = significant_df.sort_values('abs_change', ascending=False)
        significant_df.to_csv(os.path.join(args.output_dir, "significant_changes.csv"), index=False)
        
        print("\nTop 10 most significant changes:")
        cols_to_show = ['head', 'feature', 'context', args.first_subtype, args.second_subtype, 'abs_change']
        print(significant_df.head(10)[cols_to_show])
    
    # Create a summary report of all changes (not just significant ones)
    print("Creating summary of all changes...")
    
    # Global changes
    common_heads_global = set(first_global.index).intersection(set(second_global.index))
    common_features_global = set(first_global.columns).intersection(set(second_global.columns))
    
    if common_heads_global and common_features_global:
        # Convert sets to lists for pandas indexing
        common_heads_list = list(common_heads_global)
        common_features_list = list(common_features_global)
        first_filtered = first_global.loc[common_heads_list, common_features_list]
        second_filtered = second_global.loc[common_heads_list, common_features_list]
        
        # Calculate absolute differences
        diff_df = (second_filtered - first_filtered).abs()
        
        # Reshape to get head-feature pairs with their change values
        changes = []
        for head in diff_df.index:
            for feature in diff_df.columns:
                changes.append({
                    'head': head,
                    'feature': feature,
                    f'{args.first_subtype}_zscore': first_filtered.loc[head, feature],
                    f'{args.second_subtype}_zscore': second_filtered.loc[head, feature],
                    'abs_diff': diff_df.loc[head, feature],
                    'context': 'global'
                })
        
        # Add label-specific changes
        for label in common_labels:
            common_heads_label = set(first_label[label].index).intersection(set(second_label[label].index))
            common_features_label = set(first_label[label].columns).intersection(set(second_label[label].columns))
            
            if common_heads_label and common_features_label:
                # Convert sets to lists for pandas indexing
                common_heads_list = list(common_heads_label)
                common_features_list = list(common_features_label)
                first_filtered = first_label[label].loc[common_heads_list, common_features_list]
                second_filtered = second_label[label].loc[common_heads_list, common_features_list]
                
                # Calculate absolute differences
                diff_df = (second_filtered - first_filtered).abs()
                
                # Add to changes list
                for head in diff_df.index:
                    for feature in diff_df.columns:
                        changes.append({
                            'head': head,
                            'feature': feature,
                            f'{args.first_subtype}_zscore': first_filtered.loc[head, feature],
                            f'{args.second_subtype}_zscore': second_filtered.loc[head, feature],
                            'abs_diff': diff_df.loc[head, feature],
                            'context': f'label_{label}'
                        })
        
        # Create DataFrame and sort by absolute difference
        changes_df = pd.DataFrame(changes)
        changes_df = changes_df.sort_values('abs_diff', ascending=False)
        
        # Save to CSV
        changes_df.to_csv(os.path.join(args.output_dir, "zscore_changes_summary.csv"), index=False)
        
        # Print top changes
        print("\nTop 10 most changed associations:")
        cols_to_show = ['head', 'feature', 'context', f'{args.first_subtype}_zscore', 
                        f'{args.second_subtype}_zscore', 'abs_diff']
        print(changes_df.head(10)[cols_to_show])

        # Save the top 10 changes to a csv file
        changes_df.head(10)[cols_to_show].to_csv(os.path.join(args.output_dir, "top_10_changes.csv"), index=False)
        print(f"\nTop 10 changes saved to {args.output_dir}/top_10_changes.csv")
    
    print(f"\nAll plots and summaries saved to {args.output_dir}")

if __name__ == "__main__":
    main()