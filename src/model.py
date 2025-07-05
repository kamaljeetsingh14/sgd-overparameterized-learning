"""
Neural network model building module
Contains functions for creating and training neural network models
with different architectures for the double descent study
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def get_num_hidden(n):
    """
    Calculate the approximate number of hidden units for a desired number of parameters
    
    For a fully connected network with 28x28 input, H hidden units, and 10 output units:
    Total parameters = (28*28 + 1) * H + (H + 1) * 10 = 785*H + 10*H + 10 = 795*H + 10
    
    Args:
        n (int): Desired number of parameters
    
    Returns:
        float: Approximate number of hidden units needed
    """
    return (n - 10) / (785 + 10 + 1)

def create_model(num_hidden_units, train_images, train_labels, epochs=50, batch_size=64, 
                validation_split=0.2, momentum=0.95, verbose=0):
    """
    Create and train a neural network model with specified architecture
    
    Args:
        num_hidden_units (int): Number of units in the hidden layer
        train_images (np.ndarray): Training images
        train_labels (np.ndarray): Training labels
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        momentum (float): Momentum for SGD optimizer
        verbose (int): Verbosity level for training
    
    Returns:
        tuple: (training_error, validation_error, total_parameters)
    """
    if not isinstance(num_hidden_units, int) or num_hidden_units <= 0:
        raise ValueError("num_hidden_units must be a positive integer.")
    
    # Create the model architecture
    model = Sequential([
        Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784-dimensional vectors
        Dense(num_hidden_units, activation='relu'),  # Hidden layer with ReLU activation
        Dense(10, activation='softmax')  # Output layer for 10 classes
    ])
    
    # Configure SGD optimizer with momentum
    sgd = SGD(momentum=momentum)
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose
    )
    
    # Extract final training and validation errors
    train_error = history.history['loss'][-1]
    val_error = history.history['val_loss'][-1]
    total_params = model.count_params()
    
    return train_error, val_error, total_params

def create_model_with_early_stopping(num_hidden_units, train_images, train_labels, 
                                    batch_size=64, validation_split=0.2, 
                                    patience=5, max_epochs=50, verbose=0):
    """
    Create and train a model with early stopping
    
    Args:
        num_hidden_units (int): Number of units in the hidden layer
        train_images (np.ndarray): Training images
        train_labels (np.ndarray): Training labels
        batch_size (int): Batch size for training
        validation_split (float): Fraction of data to use for validation
        patience (int): Number of epochs to wait before stopping
        max_epochs (int): Maximum number of epochs
        verbose (int): Verbosity level
    
    Returns:
        tuple: (model, history, epochs_to_convergence)
    """
    # Create model
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(num_hidden_units, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compile model
    sgd = SGD(momentum=0.95)
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=verbose,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        train_images, train_labels,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=verbose
    )
    
    epochs_to_convergence = len(history.history['loss'])
    
    return model, history, epochs_to_convergence

def get_model_summary(model):
    """
    Get a summary of the model architecture
    
    Args:
        model: Keras model
    
    Returns:
        dict: Dictionary containing model information
    """
    total_params = model.count_params()
    
    # Get layer information
    layers_info = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            layers_info.append({
                'layer_index': i,
                'layer_type': type(layer).__name__,
                'units': layer.units,
                'activation': layer.activation.__name__ if hasattr(layer, 'activation') else None
            })
    
    return {
        'total_parameters': total_params,
        'layers': layers_info,
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
