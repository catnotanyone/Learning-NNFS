"""
MNIST Dataset Loader Module

This module provides functionality to load the MNIST dataset of handwritten digits.
The dataset consists of 28x28 grayscale images of handwritten digits (0-9) and their corresponding labels.

The MNIST dataset files should be present in the 'mnist-dataset' directory in IDX format:
- train-images.idx3-ubyte: Training set images
- train-labels.idx1-ubyte: Training set labels
- t10k-images.idx3-ubyte: Test set images
- t10k-labels.idx1-ubyte: Test set labels
"""

import numpy as np

def load_mnist_images(file_path: str):
    """
    Load MNIST image data from IDX file format.
    
    Args:
        file_path (str): Path to the IDX file containing image data
        
    Returns:
        numpy.ndarray: Array of shape (n_samples, height, width) containing the images
    """
    with open(file_path, 'rb') as file:
        # Read IDX file format metadata (magic number, number of images, dimensions)
        magic, num, rows, cols = np.fromfile(file, dtype=np.dtype('>i4'), count=4)
        # Read and reshape image data into a 3D array (num_images x height x width)
        images = np.fromfile(file, dtype=np.uint8).reshape(num, rows, cols)
        return images

def load_mnist_labels(file_path: str):
    """
    Load MNIST label data from IDX file format.
    
    Args:
        file_path (str): Path to the IDX file containing label data
        
    Returns:
        numpy.ndarray: Array of shape (n_samples,) containing the labels (0-9)
    """
    with open(file_path, 'rb') as f:
        # Read IDX file format metadata (magic number and number of labels)
        magic, num = np.fromfile(f, dtype=np.dtype('>i4'), count=2)
        # Read label data as unsigned 8-bit integers
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Load training dataset
train_images = load_mnist_images('mnist/mnist-dataset/train-images.idx3-ubyte')
train_labels = load_mnist_labels('mnist/mnist-dataset/train-labels.idx1-ubyte')

# Load test dataset
test_images = load_mnist_images('mnist/mnist-dataset/t10k-images.idx3-ubyte')
test_labels = load_mnist_labels('mnist/mnist-dataset/t10k-labels.idx1-ubyte')
