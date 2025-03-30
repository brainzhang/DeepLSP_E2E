import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
from typing import Tuple, Dict, Any, Union, List

DATASET_ROOT = "/mnt/d/Ubuntu/phd/Public/Datasets"
IMAGENET_ROOT = "/mnt/d/datasets/ImageNet"

def get_dataset_mean_std(dataset_name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Get the mean and standard deviation for a dataset.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', 'imagenet', 'fashionmnist')
        
    Returns:
        Tuple of mean and std tuples
    """
    # Standard normalization values
    if dataset_name == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4865, 0.4409)
        std = (0.2673, 0.2564, 0.2762)
    elif dataset_name == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif dataset_name == 'fashionmnist':
        # FashionMNIST is grayscale, but we'll convert to 3-channel
        mean = (0.2860,) * 3
        std = (0.3530,) * 3
    else:
        # Default values
        logging.warning(f"⚠️ Unknown dataset '{dataset_name}', using default normalization values")
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    
    return mean, std

def get_transforms(dataset_name: str, train: bool = True) -> transforms.Compose:
    """
    Get the transforms for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        train: Whether to get transforms for training or testing
        
    Returns:
        Composed transforms
    """
    mean, std = get_dataset_mean_std(dataset_name)
    
    if train:
        if dataset_name in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset_name == 'imagenet':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset_name == 'fashionmnist':
            transform = transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3-channel
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:
        # Test transforms - no augmentation
        if dataset_name in ['cifar10', 'cifar100']:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset_name == 'imagenet':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif dataset_name == 'fashionmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert to 3-channel
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    return transform

def get_dataset(dataset_name: str, train: bool = True) -> Any:
    """
    Get a dataset.
    
    Args:
        dataset_name: Name of the dataset
        train: Whether to get the training or testing dataset
        
    Returns:
        Dataset object
    """
    transform = get_transforms(dataset_name, train)
    
    if dataset_name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root=DATASET_ROOT, train=train, download=True, transform=transform
        )
    elif dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=DATASET_ROOT, train=train, download=True, transform=transform
        )
    elif dataset_name == 'imagenet':
        # ImageNet requires a different approach due to its size
        if train:
            data_dir = os.path.join(IMAGENET_ROOT, 'train')
        else:
            data_dir = os.path.join(IMAGENET_ROOT, 'val')
        
        if not os.path.exists(data_dir):
            logging.warning(f"⚠️ ImageNet directory not found: {data_dir}")
            logging.warning("⚠️ Please download ImageNet manually and place it in the appropriate directory")
            logging.warning("⚠️ Using CIFAR-10 as a fallback")
            return get_dataset('cifar10', train)
        
        dataset = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    elif dataset_name == 'fashionmnist':
        dataset = torchvision.datasets.FashionMNIST(
            root=DATASET_ROOT, train=train, download=True, transform=transform
        )
    else:
        logging.error(f"❌ Unknown dataset '{dataset_name}'")
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    
    return dataset

class DatasetInfo:
    """Class to store dataset information"""
    train_loader: torch.utils.data.DataLoader
    test_loader: torch.utils.data.DataLoader
    num_classes: int
    input_size: Tuple[int, int, int, int]

def get_dataloaders(dataset_name: str, batch_size: int = 128, num_workers: int = 4) -> Dict[str, Any]:
    """
    Get data loaders for training and testing.
    
    Args:
        dataset_name: Name of the dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for the data loaders
        
    Returns:
        Dictionary containing train_loader, test_loader, num_classes, and input_size
    """
    logging.info(f"📊 Loading {dataset_name} dataset with batch size {batch_size}")
    
    # Get datasets
    train_dataset = get_dataset(dataset_name, train=True)
    test_dataset = get_dataset(dataset_name, train=False)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    logging.info(f"✅ Loaded {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_classes': len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 0,
        'input_size': get_input_size(dataset_name)
    }

def get_input_size(dataset_name: str) -> Tuple[int, int, int, int]:
    """
    Get the input size for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (batch_size, channels, height, width)
    """
    if dataset_name == 'imagenet':
        return (1, 3, 224, 224)
    elif dataset_name in ['cifar10', 'cifar100']:
        return (1, 3, 32, 32)
    elif dataset_name == 'fashionmnist':
        return (1, 3, 28, 28)  # Converted to 3-channel
    else:
        return (1, 3, 32, 32)  # Default 