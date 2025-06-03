import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import backtest
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization function
def optimize_parameters(df, window_range, thres1_range):

    results = []

    # Grid search
    for window, thres1 in tqdm(list(product(window_range, thres1_range)), desc="Optimizing parameters"):
        thres2 = -thres1  # thres2 is negative of thres1
        _, metrics = backtest.calculate_metrics(df, window, thres1, thres2)
        results.append(metrics)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find parameters with maximum Sharpe Ratio
    best_params = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    
    # Save results to CSV
    results_df.to_csv('optimization_results.csv', index=False)
    
    return best_params, results_df


def plot_heatmap(results_df, metric='Sharpe Ratio'):
    # Pivot the DataFrame to create a matrix for the heatmap
    heatmap_data = results_df.pivot(index='thres1', columns='window', values=metric)
    
    # Create a figure with specified size
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap using seaborn
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='coolwarm', 
                cbar_kws={'label': metric}, center=0)
    
    # Set title and labels
    plt.title(f'{metric} Heatmap for Parameter Optimization')
    plt.xlabel('Window')
    plt.ylabel('Threshold 1')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('heatmap')
    
    # Display the plot
    plt.show()
