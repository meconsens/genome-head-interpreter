import pandas as pd
import numpy as np
import argparse
import os
import glob
from collections import defaultdict

import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.colors as mcolors
import os
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import matplotlib.patches as mpatches


def get_label_mapping(model_name):
    """Return the appropriate label mapping based on the model name."""
    label_mappings = {
        "scgpt_ms": {
            0: 'PVALB-expressing interneuron', 
            1: 'SST-expressing interneuron', 
            2: 'SV2C-expressing interneuron', 
            3: 'VIP-expressing interneuron', 
            4: 'astrocyte', 
            5: 'cortical layer 2-3 excitatory neuron A', 
            6: 'cortical layer 2-3 excitatory neuron B', 
            7: 'cortical layer 4 excitatory neuron', 
            8: 'cortical layer 5-6 excitatory neuron', 
            9: 'endothelial cell', 
            10: 'microglial cell', 
            11: 'mixed excitatory neuron', 
            12: 'mixed glial cell', 
            13: 'oligodendrocyte A', 
            14: 'oligodendrocyte C', 
            15: 'oligodendrocyte precursor cell', 
            16: 'phagocyte', 
            17: 'pyramidal neuron'
        },
        "scgpt_pancreas": {
            0: 'MHC class II', 
            1: 'PP', 
            2: 'PSC', 
            3: 'acinar', 
            4: 'alpha', 
            5: 'beta', 
            6: 'delta', 
            7: 'ductal', 
            8: 'endothelial', 
            9: 'epsilon', 
            10: 'macrophage', 
            11: 'mast', 
            12: 'schwann', 
            13: 't_cell'
        },
        "DNABERT_TATA": {
            0: 'non-TATA', 
            1: 'TATA'
        },
        "NT_TATA": {
            0: 'non-TATA', 
            1: 'TATA'
        },
        "DNABERT_enhancers": {
            0: 'non-enhancer', 
            1: 'enhancer'
        },
        "NT_enhancers": {
            0: 'non-enhancer', 
            1: 'enhancer'
        }
    }
    
    return label_mappings.get(model_name, {})


def analyze_feature_associations(full_path, model_name, subtype="finetuned", prefix="centered_", top_n=5):
    """
    Analyze feature associations for a specific model, separately for positive and negative z-scores.
    
    Args:
        full_path: Base path for data files
        model_name: Name of the model
        subtype: Model subtype (default: "finetuned")
        prefix: Prefix for z-score files (default: "centered_")
        top_n: Number of top features and heads to report (default: 5)
        
    Returns:
        Dictionary with analysis results
    """
    print(f"Analyzing {model_name} ({subtype})...")
    
    # Load z-scores
    global_zscores, label_zscores = load_zscores(full_path, model_name, subtype, prefix)
    
    if global_zscores is None:
        return None
    
    results = {
        "model_name": model_name,
        "subtype": subtype,
        "global": {},
        "label_specific": {}
    }
    
    # Analyze global z-scores
    # Separate positive and negative z-scores
    pos_global = global_zscores.copy()
    pos_global[pos_global < 0] = 0  # Keep only positive values
    
    neg_global = global_zscores.copy()
    neg_global[neg_global > 0] = 0  # Keep only negative values
    
    # Sum across all heads for each feature
    pos_feature_totals = pos_global.sum()
    neg_feature_totals = neg_global.sum()
    
    top_pos_features = pos_feature_totals.nlargest(top_n)
    top_neg_features = neg_feature_totals.nsmallest(top_n)  # Most negative values
    
    results["global"]["top_pos_features"] = {}
    results["global"]["top_neg_features"] = {}
    
    # Process positive features
    for feature, total in top_pos_features.items():
        if total > 0:  # Only include if there's a positive contribution
            # Get top heads for this feature
            feature_heads = pos_global[feature].nlargest(top_n)
            
            results["global"]["top_pos_features"][feature] = {
                "total_pos_zscore": total,
                "top_heads": {
                    head: {
                        "pos_zscore": pos_score,
                        "original_zscore": global_zscores.loc[head, feature]
                    }
                    for head, pos_score in feature_heads.items()
                    if pos_score > 0  # Only include positive contributions
                }
            }
    
    # Process negative features
    for feature, total in top_neg_features.items():
        if total < 0:  # Only include if there's a negative contribution
            # Get top heads for this feature (most negative)
            feature_heads = neg_global[feature].nsmallest(top_n)
            
            results["global"]["top_neg_features"][feature] = {
                "total_neg_zscore": total,
                "top_heads": {
                    head: {
                        "neg_zscore": neg_score,
                        "original_zscore": global_zscores.loc[head, feature]
                    }
                    for head, neg_score in feature_heads.items()
                    if neg_score < 0  # Only include negative contributions
                }
            }
    
    # Analyze label-specific z-scores
    for label, label_df in label_zscores.items():
        # Separate positive and negative z-scores
        pos_label = label_df.copy()
        pos_label[pos_label < 0] = 0  # Keep only positive values
        
        neg_label = label_df.copy()
        neg_label[neg_label > 0] = 0  # Keep only negative values
        
        # Sum across all heads for each feature
        pos_feature_totals = pos_label.sum()
        neg_feature_totals = neg_label.sum()
        
        top_pos_features = pos_feature_totals.nlargest(top_n)
        top_neg_features = neg_feature_totals.nsmallest(top_n)  # Most negative values
        
        results["label_specific"][label] = {
            "pos_features": {},
            "neg_features": {}
        }
        
        # Process positive features
        for feature, total in top_pos_features.items():
            if total > 0:  # Only include if there's a positive contribution
                # Get top heads for this feature
                feature_heads = pos_label[feature].nlargest(top_n)
                
                results["label_specific"][label]["pos_features"][feature] = {
                    "total_pos_zscore": total,
                    "top_heads": {
                        head: {
                            "pos_zscore": pos_score,
                            "original_zscore": label_df.loc[head, feature]
                        }
                        for head, pos_score in feature_heads.items()
                        if pos_score > 0  # Only include positive contributions
                    }
                }
        
        # Process negative features
        for feature, total in top_neg_features.items():
            if total < 0:  # Only include if there's a negative contribution
                # Get top heads for this feature (most negative)
                feature_heads = neg_label[feature].nsmallest(top_n)
                
                results["label_specific"][label]["neg_features"][feature] = {
                    "total_neg_zscore": total,
                    "top_heads": {
                        head: {
                            "neg_zscore": neg_score,
                            "original_zscore": label_df.loc[head, feature]
                        }
                        for head, neg_score in feature_heads.items()
                        if neg_score < 0  # Only include negative contributions
                    }
                }
    
    return results


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
            print(f"No z-score file found for {model_name}/{subtype}. Skipping.")
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

def find_all_models(full_path):
    """
    Find all model directories in the z_scores directory.
    
    Args:
        full_path: Base path for data files
        
    Returns:
        List of model names
    """
    model_paths = glob.glob(f"{full_path}/data/z_scores/*")
    return [os.path.basename(path) for path in model_paths]



def plot_feature_violin_distributions(all_results, output_dir):
    """
    Create a single plot with violin plots showing the distribution of z-scores
    for each head, per feature, with different colors for each label.
    
    Args:
        all_results: List of dictionaries with analysis results
        output_dir: Directory to save the plot
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap
    from scipy import stats
    
    # Create DataFrame for plotting
    plot_data = []
    
    # Process all results
    for results in all_results:
        if results is None:
            continue
        
        model_name = results["model_name"]
        subtype = results["subtype"]
        
        # Load original z-score data to get distributions for each head
        global_zscores, label_zscores = load_zscores(os.path.dirname(output_dir), model_name, subtype)
        
        if global_zscores is None:
            continue
        
        # Get label mapping for this model
        label_mapping = get_label_mapping(model_name)
        
        # Skip global z-scores and only process label-specific z-scores
        for label, label_df in label_zscores.items():
            # Try to convert label to integer for mapping if it's not already
            try:
                label_idx = int(label)
                label_name = label_mapping.get(label_idx, f"label_{label}")
            except (ValueError, TypeError):
                label_name = f"label_{label}"
                
            for feature in label_df.columns:
                for head in label_df.index:
                    z_score = label_df.loc[head, feature]
                    plot_data.append({
                        "model": model_name,
                        "subtype": subtype,
                        "context": label_name,
                        "original_label": label,
                        "feature": feature,
                        "head": head,
                        "z_score": z_score
                    })
    
    # Create DataFrame
    df = pd.DataFrame(plot_data)
    
    if df.empty:
        print("No data to plot")
        return
    
    # Process each model and subtype
    for (model, subtype), model_df in df.groupby(['model', 'subtype']):
        print(f"Creating violin plot for {model} ({subtype})...")
        
        # Get unique features and contexts
        features = sorted(model_df['feature'].unique())
        contexts = sorted(model_df['context'].unique())
        
        # Calculate features with most different distributions between labels
        feature_diff_scores = {}
        
        # For each feature, calculate how different the distributions are between labels
        for feature in features:
            feature_df = model_df[model_df['feature'] == feature]
            
            # Skip if we don't have data for at least 2 contexts
            if len(feature_df['context'].unique()) < 2:
                continue
            
            # Calculate pairwise statistical difference between distributions
            context_pairs = []
            for i, context1 in enumerate(contexts):
                for context2 in contexts[i+1:]:
                    context1_scores = feature_df[feature_df['context'] == context1]['z_score'].values
                    context2_scores = feature_df[feature_df['context'] == context2]['z_score'].values
                    
                    # Skip if either context doesn't have enough data
                    if len(context1_scores) < 2 or len(context2_scores) < 2:
                        continue
                    
                    # Calculate KS statistic (measure of distribution difference)
                    ks_stat, _ = stats.ks_2samp(context1_scores, context2_scores)
                    context_pairs.append(ks_stat)
            
            # Average of KS statistics for this feature (higher means more different)
            if context_pairs:
                feature_diff_scores[feature] = np.mean(context_pairs)
        
        # Get top 5 features with most different distributions (if available)
        top_diff_features = []
        if feature_diff_scores:
            top_diff_features = sorted(feature_diff_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            top_diff_features = [f[0] for f in top_diff_features]
        
        # Calculate features with greatest magnitude z-scores
        feature_magnitude_scores = {}
        for feature in features:
            feature_df = model_df[model_df['feature'] == feature]
            # Use absolute values to measure magnitude regardless of direction
            mean_abs_zscore = np.mean(np.abs(feature_df['z_score'].values))
            feature_magnitude_scores[feature] = mean_abs_zscore
        
        # Get top 5 features with greatest magnitude z-scores
        top_magnitude_features = []
        if feature_magnitude_scores:
            top_magnitude_features = sorted(feature_magnitude_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            top_magnitude_features = [f[0] for f in top_magnitude_features]
        
        # Define base colors for labels
        base_colors = {
            'non-TATA': '#8B0000',        # Dark red
            'TATA': '#006400',            # Dark green
            'non-enhancer': '#00008B',    # Dark blue
            'enhancer': '#8B008B',        # Dark magenta
            # Add more base colors for specific cell types
            'PVALB-expressing interneuron': '#E41A1C',      # Red
            'SST-expressing interneuron': '#377EB8',        # Blue
            'VIP-expressing interneuron': '#4DAF4A',        # Green
            'astrocyte': '#984EA3',                         # Purple
            'oligodendrocyte A': '#FF7F00',                 # Orange
            'oligodendrocyte C': '#FFFF33',                 # Yellow
            'beta': '#A65628',                              # Brown
            'alpha': '#F781BF',                             # Pink
            'delta': '#999999',                             # Grey
        }
        
        # Generate colors for contexts not in base_colors
        label_colors = {}
        for i, context in enumerate(contexts):
            if context in base_colors:
                label_colors[context] = base_colors[context]
            else:
                # Generate a color based on position in list
                h = (i * 0.1) % 1.0  # Hue (cycle through colors)
                s = 0.7              # Saturation
                v = 0.6              # Value (darker)
                
                import matplotlib.colors as mcolors
                rgb = mcolors.hsv_to_rgb((h, s, v))
                label_colors[context] = mcolors.rgb2hex(rgb)
        
        # Create plots for all features (Full Plot)
        print(f"  Creating full feature plot...")
        all_features = features
        create_violin_plot(model_df, all_features, contexts, label_colors, 
                           os.path.join(output_dir, f"{model}_{subtype}_feature_violin_distributions.png"),
                           model, subtype, "All Features")
        
        # Create plot for top differentiating features (if available)
        if top_diff_features:
            print(f"  Creating top differentiating features plot...")
            create_violin_plot(model_df, top_diff_features, contexts, label_colors, 
                               os.path.join(output_dir, f"{model}_{subtype}_top_diff_features.png"),
                               model, subtype, "Top 5 Differentiating Features")
        
        # Create plot for top magnitude features (if available)
        if top_magnitude_features:
            print(f"  Creating top magnitude features plot...")
            create_violin_plot(model_df, top_magnitude_features, contexts, label_colors, 
                               os.path.join(output_dir, f"{model}_{subtype}_top_magnitude_features.png"),
                               model, subtype, "Top 5 Features by Z-score Magnitude")

def create_violin_plot(df, features, contexts, label_colors, output_path, model, subtype, subtitle):
    """
    Helper function to create a violin plot with the given features and contexts.
    
    Args:
        df: DataFrame with the data
        features: List of features to plot
        contexts: List of contexts (labels)
        label_colors: Dictionary mapping contexts to colors
        output_path: Path to save the plot
        model: Model name for the title
        subtype: Subtype name for the title
        subtitle: Subtitle for the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    # Determine the figure size based on the number of features
    fig_width = max(15, len(features) * 0.8)
    fig_height = 10
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set position index for plotting
    pos = 0
    x_positions = []
    x_labels = []
    
    # Plot each feature
    for feature_idx, feature in enumerate(features):
        feature_df = df[df['feature'] == feature]
        
        if feature_df.empty:
            continue
        
        # Track feature position
        feature_pos = pos
        x_positions.append(feature_pos + len(contexts) / 2 - 0.5)
        x_labels.append(feature)
        
        # Plot each context for this feature
        for context_idx, context in enumerate(contexts):
            context_df = feature_df[feature_df['context'] == context]
            
            if context_df.empty:
                pos += 1
                continue
            
            # Extract z-scores for this feature and context
            z_scores = context_df['z_score'].values
            
            # Plot violin
            if len(z_scores) > 1:  # Violin plot requires at least 2 data points
                parts = plt.violinplot(z_scores, positions=[pos], showmeans=True, showextrema=True)
                
                # Set color based on context
                for pc in parts['bodies']:
                    pc.set_facecolor(label_colors[context])
                    pc.set_alpha(0.7)
                
                parts['cmeans'].set_color(label_colors[context])
                parts['cmins'].set_color(label_colors[context])
                parts['cmaxes'].set_color(label_colors[context])
                parts['cbars'].set_color(label_colors[context])
            else:
                # If only one data point, plot a marker
                plt.plot([pos], z_scores, 'o', color=label_colors[context], markersize=8)
            
            pos += 1
        
        # Add a small gap between features
        pos += 0.5
    
    # Set x-ticks and labels
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    
    # Add gridlines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Set labels and title
    plt.ylabel('Z-score Value')
    plt.title(f'Z-score Distributions by Feature for {model} ({subtype})\n{subtitle}', fontsize=14)
    
    # Create legend for contexts
    legend_patches = [mpatches.Patch(color=color, label=context) 
                      for context, color in label_colors.items() 
                      if context in contexts]
    
    # If there are too many labels, create a compact legend
    if len(legend_patches) > 10:
        plt.legend(handles=legend_patches, loc='upper right', fontsize='small', ncol=2)
    else:
        plt.legend(handles=legend_patches, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to {output_path}")

def create_violin_plot(df, features, contexts, label_colors, output_path, model, subtype, subtitle):
    """
    Helper function to create a violin plot with the given features and contexts.
    
    Args:
        df: DataFrame with the data
        features: List of features to plot
        contexts: List of contexts (labels)
        label_colors: Dictionary mapping contexts to colors
        output_path: Path to save the plot
        model: Model name for the title
        subtype: Subtype name for the title
        subtitle: Subtitle for the plot
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    
    # Determine the figure size based on the number of features
    fig_width = max(15, len(features) * 0.8)
    fig_height = 10
    
    # Create figure
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set position index for plotting
    pos = 0
    x_positions = []
    x_labels = []
    
    # Plot each feature
    for feature_idx, feature in enumerate(features):
        feature_df = df[df['feature'] == feature]
        
        if feature_df.empty:
            continue
        
        # Track feature position
        feature_pos = pos
        x_positions.append(feature_pos + len(contexts) / 2 - 0.5)
        x_labels.append(feature)
        
        # Plot each context for this feature
        for context_idx, context in enumerate(contexts):
            context_df = feature_df[feature_df['context'] == context]
            
            if context_df.empty:
                pos += 1
                continue
            
            # Extract z-scores for this feature and context
            z_scores = context_df['z_score'].values
            
            # Plot violin
            if len(z_scores) > 1:  # Violin plot requires at least 2 data points
                parts = plt.violinplot(z_scores, positions=[pos], showmeans=True, showextrema=True)
                
                # Set color based on context
                for pc in parts['bodies']:
                    pc.set_facecolor(label_colors[context])
                    pc.set_alpha(0.7)
                
                parts['cmeans'].set_color(label_colors[context])
                parts['cmins'].set_color(label_colors[context])
                parts['cmaxes'].set_color(label_colors[context])
                parts['cbars'].set_color(label_colors[context])
            else:
                # If only one data point, plot a marker
                plt.plot([pos], z_scores, 'o', color=label_colors[context], markersize=8)
            
            pos += 1
        
        # Add a small gap between features
        pos += 0.5
    
    # Set x-ticks and labels
    plt.xticks(x_positions, x_labels, rotation=45, ha='right')
    
    # Add gridlines
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Set labels and title
    plt.ylabel('Z-score Value')
    plt.title(f'Z-score Distributions by Feature for {model} ({subtype})\n{subtitle}', fontsize=14)
    
    # Create legend for contexts
    legend_patches = [mpatches.Patch(color=color, label=context) 
                      for context, color in label_colors.items() 
                      if context in contexts]
    plt.legend(handles=legend_patches, loc='upper right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot to {output_path}")


# Update the main function to only generate violin plots
def main():
    parser = argparse.ArgumentParser(description='Generate violin plots for feature-head associations')
    parser.add_argument("--full_path", default="/home/mica/genome-head-interpreter/preprocessing/", type=str)
    parser.add_argument("--output_dir", default="/home/mica/genome-head-interpreter/preprocessing/feature_head_associations", type=str)
    parser.add_argument("--prefix", default="centered_", type=str)
    parser.add_argument("--model", default=None, type=str,
                        help="Specific model to analyze (analyze all models if not specified)")
    parser.add_argument("--subtype", default="finetuned", type=str,
                        help="Model subtype to analyze (default: finetuned)")
    parser.add_argument("--top_n", default=5, type=int,
                        help="Number of top features and heads to report (default: 5)")
    
    args = parser.parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    if args.model:
        # Analyze a specific model
        results = analyze_feature_associations(
            args.full_path, 
            args.model, 
            args.subtype, 
            args.prefix,
            args.top_n
        )
        
        if results:
            all_results.append(results)
    else:
        # Analyze all models
        models = find_all_models(args.full_path)
        
        for model in models:
            results = analyze_feature_associations(
                args.full_path, 
                model, 
                args.subtype, 
                args.prefix,
                args.top_n
            )
            
            if results:
                all_results.append(results)
    
    # Create the violin plot visualization
    plot_feature_violin_distributions(all_results, output_dir)
    
    print(f"\nViolin plots created and saved to {output_dir}/")

if __name__ == "__main__":
    main()