"""
Main script for Double Descent Phenomenon Analysis
This script demonstrates the double descent phenomenon in neural networks
by training models with varying numbers of parameters on the MNIST dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_mnist_data
from model_builder import create_model, get_num_hidden
from convergence_analysis import analyze_convergence_rates, plot_convergence_comparison
from spectral_analysis import analyze_spectral_properties
from visualization import plot_double_descent_curve, plot_loss_vs_iterations

def main():
    """
    Main function to execute the double descent analysis
    """
    print("Starting Double Descent Analysis...")
    
    # Load and preprocess MNIST data
    print("Loading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Reduce dataset size to 4000 samples (aligned with double-descent paper)
    train_images = train_images[:4000]
    train_labels = train_labels[:4000]
    print(f"Using {len(train_images)} training samples")
    
    # Define parameter ranges for double descent curve
    interpolation_threshold = len(train_images) * 10  # 40,000 parameters
    interp_thresh_H = round(get_num_hidden(interpolation_threshold))
    
    # Parameters to test (number of hidden units)
    params = [4, 8, 16, 32] + list(range(interp_thresh_H-8, interp_thresh_H+8, 2)) + [64, 76, 88, 100]
    
    print(f"Testing parameters: {params}")
    print(f"Interpolation threshold: {interpolation_threshold} parameters")
    
    # Generate double descent curve
    print("\nGenerating double descent curve...")
    error_results = []
    
    for i, num_params in enumerate(params):
        print(f"Training model {i+1}/{len(params)} with {num_params} hidden units...")
        train_error, val_error, total_params = create_model(num_params, train_images, train_labels)
        error_results.append({
            'number of parameters': total_params,
            'Training Error': train_error,
            'Validation Error': val_error,
            'Hidden Units': num_params
        })
    
    # Convert to DataFrame and save results
    error_df = pd.DataFrame(error_results)
    error_df.to_csv('double_descent_results.csv', index=False)
    print("Results saved to 'double_descent_results.csv'")
    
    # Plot double descent curve
    print("\nPlotting double descent curve...")
    plot_double_descent_curve(error_df, len(train_images))
    
    # Analyze convergence rates for different batch sizes
    print("\nAnalyzing convergence rates...")
    batch_sizes = [2**i for i in range(1, 8)]  # [2, 4, 8, 16, 32, 64, 128]
    
    convergence_results = analyze_convergence_rates(
        train_images, train_labels, batch_sizes, 
        under_param_size=32, over_param_size=64, num_runs=3
    )
    
    # Plot convergence comparison
    plot_convergence_comparison(convergence_results)
    
    # Analyze spectral properties
    print("\nAnalyzing spectral properties...")
    spectral_results = analyze_spectral_properties(train_images, train_labels)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("- double_descent_results.csv")
    print("- double_descent_curve.png")
    print("- convergence_analysis.png")
    print("- spectral_analysis_results.txt")

if __name__ == "__main__":
    main()
