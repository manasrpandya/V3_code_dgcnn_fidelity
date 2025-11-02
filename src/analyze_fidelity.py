import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_model
from torch.utils.data import DataLoader
import pandas as pd
import os

def collect_fidelity_scores(model, data_loader, use_cuda=False):
    """Collect fidelity scores from model's forward passes."""
    model.eval()
    all_weights = []
    
    with torch.no_grad():
        for batch_x, _, _ in data_loader:
            _, text, audio, vision = batch_x
            
            if use_cuda:
                text = text.cuda()
                audio = audio.cuda()
                vision = vision.cuda()
            
            # Forward pass
            _, _ = model(text, audio, vision)
            
            # Get the current weights from the softmax layer
            weights = model.weight_softmax(model.modality_weights).cpu().numpy()
            all_weights.append(weights)
    
    return np.array(all_weights)

def analyze_fidelity(args, train_loader):
    """Analyze and visualize fidelity scores."""
    # Create output directory
    os.makedirs('fidelity_analysis', exist_ok=True)
    
    # Set the style for all plots - using a built-in style instead of seaborn
    plt.style.use('bmh')  # Alternative built-in style that looks professional
    
    # Set custom color palette similar to seaborn's
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    # Set font sizes
    SMALL_SIZE = 12
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 16
    
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=BIGGER_SIZE)
    plt.rc('axes', labelsize=MEDIUM_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=BIGGER_SIZE)
    
    # Load and prepare data
    model = load_model(args, name=args.name)
    if args.use_cuda:
        model = model.cuda()
    
    fidelity_scores = collect_fidelity_scores(model, train_loader, args.use_cuda)
    mean_scores = np.mean(fidelity_scores, axis=0)
    std_scores = np.std(fidelity_scores, axis=0)
    modalities = ['Text', 'Audio', 'Video']
    
    # 1. Enhanced bar plot for average fidelity scores
    plt.figure(figsize=(12, 8))
    bars = plt.bar(modalities, mean_scores, color=['#2ecc71', '#3498db', '#e74c3c'])
    plt.errorbar(modalities, mean_scores, yerr=std_scores, fmt='none', color='black', capsize=5, capthick=2, elinewidth=2)
    
    plt.title('Average Fidelity Scores per Modality\nContribution of Each Modality to Final Prediction', pad=20)
    plt.ylabel('Fidelity Score', labelpad=15)
    plt.ylim(0, max(mean_scores) + max(std_scores) + 0.1)
    
    # Add value labels on top of bars - now showing only the mean scores
    for bar, score, std in zip(bars, mean_scores, std_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{score:.3f}',  # Removed the ± std part
                ha='center', va='bottom', fontsize=12)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.savefig('fidelity_analysis/avg_fidelity_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Enhanced violin plot
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=pd.DataFrame(fidelity_scores, columns=modalities),
                  palette=['#2ecc71', '#3498db', '#e74c3c'])
    plt.title('Distribution of Fidelity Scores Across Modalities\nKernel Density Estimation of Score Distribution', pad=20)
    plt.ylabel('Fidelity Score', labelpad=15)
    
    # Add mean lines
    plt.axhline(y=mean_scores[0], color='#27ae60', linestyle='--', alpha=0.5)
    plt.axhline(y=mean_scores[1], color='#2980b9', linestyle='--', alpha=0.5)
    plt.axhline(y=mean_scores[2], color='#c0392b', linestyle='--', alpha=0.5)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.savefig('fidelity_analysis/fidelity_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Enhanced temporal evolution plot
    plt.figure(figsize=(15, 8))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for i, (modality, color) in enumerate(zip(modalities, colors)):
        plt.plot(fidelity_scores[:, i], label=modality, color=color, 
                linewidth=2, alpha=0.7)
        plt.axhline(y=mean_scores[i], color=color, linestyle='--', 
                   alpha=0.5, label=f'{modality} Mean')
    
    plt.title('Temporal Evolution of Fidelity Scores\nHow Modality Importance Changes Across Samples', pad=20)
    plt.xlabel('Sample Index', labelpad=15)
    plt.ylabel('Fidelity Score', labelpad=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig('fidelity_analysis/temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Enhanced correlation heatmap with validation
    plt.figure(figsize=(10, 8))
    
    # Validate fidelity scores
    if np.isnan(fidelity_scores).any():
        print("Warning: NaN values detected in fidelity scores!")
        # Replace NaN with 0 for visualization
        fidelity_scores = np.nan_to_num(fidelity_scores, 0)
    
    if np.all(fidelity_scores == 0):
        print("Warning: All fidelity scores are zero!")
        return None
    
    # Calculate correlation with validated data
    correlation = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                correlation[i,j] = 1.0
            else:
                valid_mask = ~(np.isnan(fidelity_scores[:,i]) | np.isnan(fidelity_scores[:,j]))
                if np.sum(valid_mask) > 1:  # Need at least 2 valid points
                    correlation[i,j] = np.corrcoef(fidelity_scores[valid_mask,i], 
                                                 fidelity_scores[valid_mask,j])[0,1]
                else:
                    correlation[i,j] = 0
    
    # Create mask for lower triangle
    mask = np.tril(np.ones_like(correlation), k=0)
    
    # Plot heatmap with full correlation matrix
    sns.heatmap(correlation, annot=True, cmap='RdYlBu_r', fmt='.3f',
                xticklabels=modalities, yticklabels=modalities,
                mask=mask, square=True, cbar_kws={'label': 'Correlation Coefficient'},
                annot_kws={'size': 12})
    
    plt.title('Correlation between Modality Fidelity Scores\nHow Different Modalities Interact', pad=20)
    plt.tight_layout()
    plt.savefig('fidelity_analysis/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    results = {
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'correlations': correlation
    }
    
    return results

def print_analysis_results(results):
    """Print numerical analysis results."""
    if results is None:
        print("\nError: Analysis results are not available.")
        return
        
    modalities = ['Text', 'Audio', 'Video']
    
    print("\n=== Fidelity Analysis Results ===")
    print("\nMean Fidelity Scores:")
    for modality, score in zip(modalities, results['mean_scores']):
        print(f"{modality}: {score:.3f} ± {results['std_scores'][modalities.index(modality)]:.3f}")
    
    print("\nModality Correlations:")
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities[i+1:], i+1):
            corr = results['correlations'][i,j]
            if np.isnan(corr):
                print(f"{mod1}-{mod2}: No valid correlation (insufficient data)")
            else:
                print(f"{mod1}-{mod2}: {corr:.3f}")

# Make sure these functions are explicitly in __all__
__all__ = ['analyze_fidelity', 'print_analysis_results', 'collect_fidelity_scores']

# Add at top of file after imports
def __getattr__(name):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
