"""
Data loading and preprocessing module for MNIST dataset
Handles loading, normalization, and one-hot encoding of the MNIST dataset
"""

import tensorflow as tf
import tensorflow.keras.datasets.mnist as mnist
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_mnist_data(normalize=True, one_hot=True):
    """
    Load and preprocess MNIST dataset
    
    Args:
        normalize (bool): Whether to normalize pixel values to [0,1]
        one_hot (bool): Whether to one-hot encode labels
    
    Returns:
        tuple: (train_images, train_labels, test_images, test_labels)
    """
    print("Downloading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    print(f"Original shapes:")
    print(f"  Training images: {train_images.shape}")
    print(f"  Training labels: {train_labels.shape}")
    print(f"  Testing images: {test_images.shape}")
    print(f"  Testing labels: {test_labels.shape}")
    
    if normalize:
        # Normalize pixel values to be between 0 and 1
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        print("Normalized pixel values to [0,1]")
    
    if one_hot:
        # One-hot encode the labels
        train_labels = to_categorical(train_labels, 10)
        test_labels = to_categorical(test_labels, 10)
        print("Applied one-hot encoding to labels")
    
    return train_images, train_labels, test_images, test_labels

def create_subset(images, labels, subset_size):
    """
    Create a subset of the dataset
    
    Args:
        images (np.ndarray): Input images
        labels (np.ndarray): Input labels
        subset_size (int): Size of the subset
    
    Returns:
        tuple: (subset_images, subset_labels)
    """
    if subset_size > len(images):
        print(f"Warning: Requested subset size {subset_size} is larger than dataset size {len(images)}")
        return images, labels
    
    indices = np.random.choice(len(images), subset_size, replace=False)
    return images[indices], labels[indices]

def get_data_info(images, labels):
    """
    Print information about the dataset
    
    Args:
        images (np.ndarray): Input images
        labels (np.ndarray): Input labels
    """
    print(f"Dataset Information:")
    print(f"  Number of samples: {len(images)}")
    print(f"  Image shape: {images.shape[1:]}")
    print(f"  Label shape: {labels.shape[1:] if len(labels.shape) > 1 else 'scalar'}")
    print(f"  Pixel value range: [{images.min():.3f}, {images.max():.3f}]")
    
    if len(labels.shape) == 1:
        print(f"  Number of classes: {len(np.unique(labels))}")
    else:
        print(f"  Number of classes: {labels.shape[1]}")
