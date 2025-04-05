import torch
import torchvision
import numpy as np
from typing import Tuple, Union, Optional
from torch.utils.data import TensorDataset, DataLoader, Subset

import torch.utils.data as data
import torchvision.transforms as transforms


def create_gaussian_clusters(
    n_samples: int = 600,
    centers: int = 3,
    n_features: int = 2,
    cluster_std: float = 0.5,
    random_state: int = 42
) -> TensorDataset:
    """
    Create toy dataset with Gaussian clusters
    
    Args:
        n_samples: Number of samples
        centers: Number of centers/clusters
        n_features: Number of features
        cluster_std: Standard deviation of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        TensorDataset containing the generated data
    """
    np.random.seed(random_state)
    
    # Generate samples per cluster
    samples_per_cluster = n_samples // centers
    
    # Create centers for each cluster
    cluster_centers = np.random.uniform(-10, 10, size=(centers, n_features))
    
    # Generate data points
    X = []
    y = []
    
    for i in range(centers):
        # Generate points around each center
        cluster_data = np.random.normal(
            loc=cluster_centers[i],
            scale=cluster_std,
            size=(samples_per_cluster, n_features)
        )
        X.append(cluster_data)
        y.append(np.full(samples_per_cluster, i))
    
    # Combine data from all clusters
    X = np.vstack(X).astype(np.float32)
    y = np.hstack(y).astype(np.int64)
    
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    return TensorDataset(X_tensor, y_tensor)


def get_mnist_dataset(root: str = './data', train: bool = True) -> data.Dataset:
    """
    Load the MNIST dataset.
    
    Args:
        root: Directory where the datasets are saved
        train: True for training data, False for test data
        
    Returns:
        tuple: (dataset, input_dim, n_channels)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.MNIST(root=root, train=train, download=True, transform=transform)

    
    return dataset


def get_mnist_subset(root: str = './data', train: bool = True) -> data.Dataset:
    """
    Load the first 3 digits of MNIST (0, 1, 2).
    
    Args:
        root: Directory where the datasets are saved
        train: True for training data, False for test data
        
    Returns:
        tuple: (dataset, input_dim, n_channels)
    """
    dataset = get_mnist_dataset(root, train)
    
    # Extract indices where labels are 0, 1 or 2
    indices = [i for i, (_, label) in enumerate(dataset) if label <= 2]
    
    # Create a subset with only these indices
    subset = Subset(dataset, indices)
    
    return subset


def get_gaussian_dataset(
    n_samples: int = 600, 
    features: int = 2,
    centers: int = 3
) -> data.Dataset:
    """
    Create the Gaussian clusters dataset.
    
    Args:
        n_samples: Number of samples
        features: Number of features per sample
        centers: Number of clusters
        
    Returns:
        tuple: (dataset, input_dim, n_channels)
    """
    dataset = create_gaussian_clusters(
        n_samples=n_samples,
        centers=centers,
        n_features=features
    )
    
    return dataset

def get_dataloader(
    config: object,
    dataset_name: Optional[str] = None
) -> DataLoader:
    """
    Get the appropriate dataloader based on configuration.
    
    Args:
        config: Configuration object with parameters
        dataset_name: Override dataset name from config
        
    Returns:
        tuple: (dataloader)
    """
    # Use dataset name from config if not provided
    if dataset_name is None:
        dataset_name = getattr(config, 'dataset_name', '')
    
    batch_size = getattr(config, 'batch_size', 64)
    train = getattr(config, 'train', True)
    data_root = getattr(config, 'data_root', './data')
    
    # Select dataset based on name
    if dataset_name.lower() == 'mnist':
        dataset = get_mnist_dataset(data_root, train)
    elif dataset_name.lower() == 'mnist3':
        dataset = get_mnist_subset(data_root, train)
    elif dataset_name.lower() == 'gaussian':
        n_samples = getattr(config, 'n_samples', 600)
        features = getattr(config, 'features', 2)
        centers = getattr(config, 'centers', 3)
        dataset= get_gaussian_dataset(n_samples, features, centers)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=getattr(config, 'num_workers', 2),
        pin_memory=getattr(config, 'pin_memory', True)
    )
    
    return dataloader