"""
Convergence analysis module
Analyzes how convergence rates vary with batch size in under-parameterized
and over-parameterized regimes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import Callback
from model_builder import create_model_with_early_stopping

class StopAfterIterations(Callback):
    """
    Custom callback to stop training after a specified number of iterations
    and track loss after each batch
    """
    def __init__(self, max_iterations):
        super(StopAfterIterations, self).__init__()
        self.max_iterations = max_iterations
        self.iterations = 0
        self.accuracy = []
        self.loss = []

    def on_batch_end(self, batch, logs=None):
        """Record loss and accuracy after each batch"""
        self.accuracy.append(logs.get('accuracy'))
        self.loss.append(logs.get('loss'))
        self.iterations += 1
        
        if self.iterations >= self.max_iterations:
            self.model.stop_training = True
            if hasattr(self, 'verbose') and self.verbose:
                print(f"Stopped training after {self.max_iterations} iterations.")

    def on_epoch_begin(self, epoch, logs=None):
        """Reset iteration counter at the beginning of each epoch"""
        pass

def analyze_convergence_rates(train_images, train_labels, batch_sizes, 
                            under_param_size=32, over_param_size=64, num_runs=3):
    """
    Analyze convergence rates for different batch sizes in under- and over-parameterized regimes
    
    Args:
        train_images (np.ndarray): Training images
        train_labels (np.ndarray): Training labels
        batch_sizes (list): List of batch sizes to test
        under_param_size (int): Number of hidden units for under-parameterized model
        over_param_size (int): Number of hidden units for over-parameterized model
        num_runs (int): Number of runs to average over
    
    Returns:
        pd.DataFrame: Results containing convergence information
    """
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        for run in range(num_runs):
            # Under-parameterized model
            _, _, epochs_under = create_model_with_early_stopping(
                under_param_size, train_images, train_labels, 
                batch_size=batch_size, verbose=0
            )
            
            # Over-parameterized model
            _, _, epochs_over = create_model_with_early_stopping(
                over_param_size, train_images, train_labels, 
                batch_size=batch_size, verbose=0
            )
            
            # Calculate number of iterations (batches processed)
            val_pct = 0.2
            num_samples = len(train_images) * (1 - val_pct)
            batches_per_epoch = math.ceil(num_samples / batch_size)
            
            iterations_under = epochs_under * batches_per_epoch
            iterations_over = epochs_over * batches_per_epoch
            
            results.append({
                'batch_size': batch_size,
                'run': run,
                'epochs_under': epochs_under,
                'epochs_over': epochs_over,
                'iterations_under': iterations_under,
                'iterations_over': iterations_over
            })
    
    return pd.DataFrame(results)

def plot_convergence_comparison(convergence_df):
    """
    Plot convergence comparison between under- and over-parameterized models
    
    Args:
        convergence_df (pd.DataFrame): Results from analyze_convergence_rates
    """
    # Calculate mean values for each batch size
    summary = convergence_df.groupby('batch_size').mean().reset_index()
    
    # Create subplot for epochs and iterations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot epochs to convergence
    ax1.plot(summary['batch_size'], summary['epochs_under'], 
             'o-', label='Under-parameterized', linewidth=2, markersize=8)
    ax1.plot(summary['batch_size'], summary['epochs_over'], 
             's-', label='Over-parameterized', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Epochs to Convergence')
    ax1.set_title('Epochs to Convergence vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot iterations to convergence
    ax2.plot(summary['batch_size'], summary['iterations_under'], 
             'o-', label='Under-parameterized', linewidth=2, markersize=8)
    ax2.plot(summary['batch_size'], summary['iterations_over'], 
             's-', label='Over-parameterized', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('Iterations to Convergence vs Batch Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nConvergence Analysis Summary:")
    print("=" * 50)
    for _, row in summary.iterrows():
        print(f"Batch Size: {row['batch_size']}")
        print(f"  Under-parameterized: {row['epochs_under']:.1f} epochs, {row['iterations_under']:.0f} iterations")
        print(f"  Over-parameterized:  {row['epochs_over']:.1f} epochs, {row['iterations_over']:.0f} iterations")
        print()

def analyze_loss_curves(train_images, train_labels, batch_sizes, max_iterations=500):
    """
    Analyze loss curves for different batch sizes
    
    Args:
        train_images (np.ndarray): Training images
        train_labels (np.ndarray): Training labels
        batch_sizes (list): List of batch sizes to test
        max_iterations (int): Maximum number of iterations to train
    
    Returns:
        tuple: (under_results, over_results) - DataFrames with loss curves
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Flatten
    from tensorflow.keras.optimizers import SGD
    
    under_results = {}
    over_results = {}
    
    for batch_size in batch_sizes:
        print(f"Analyzing loss curves for batch size: {batch_size}")
        
        # Under-parameterized model
        under_model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        under_sgd = SGD(momentum=0.95)
        under_model.compile(optimizer=under_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Over-parameterized model
        over_model = Sequential([
            Flatten(input_shape=(28, 28)),
            Dense(64, activation='relu'),
            Dense(10, activation='softmax')
        ])
        over_sgd = SGD(momentum=0.95)
        over_model.compile(optimizer=over_sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train with custom callback
        under_callback = StopAfterIterations(max_iterations)
        over_callback = StopAfterIterations(max_iterations)
        
        under_model.fit(train_images, train_labels, 
                       epochs=max_iterations, batch_size=batch_size,
                       validation_split=0.2, callbacks=[under_callback], verbose=0)
        
        over_model.fit(train_images, train_labels, 
                      epochs=max_iterations, batch_size=batch_size,
                      validation_split=0.2, callbacks=[over_callback], verbose=0)
        
        under_results[batch_size] = under_callback.loss
        over_results[batch_size] = over_callback.loss
    
    return under_results, over_results
