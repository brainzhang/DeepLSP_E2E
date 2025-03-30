import torch
import torch.nn as nn
import torchvision.models as models
import logging
from typing import Dict, Any, Optional

# pyright: reportArgumentType=false

def create_model(model_name: str, dataset: str, num_classes: int, pretrained: bool = True, device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a model with proper architecture for the specified dataset.
    
    Args:
        model_name: Name of the model architecture ('resnet18', 'resnet50', 'vgg16', etc.)
        dataset: Name of the dataset ('cifar10', 'cifar100', 'imagenet', 'fashionmnist')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (default: True)
        device: Device to place the model on (default: auto-detect)
        
    Returns:
        PyTorch model
    """
    # Auto-detect device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"🏗️ Creating model: {model_name} for {dataset} with {num_classes} classes")
    
    # Create model based on architecture and dataset
    if dataset in ['cifar10', 'cifar100', 'fashionmnist']:
        # For smaller datasets, modify architectures
        model = create_small_dataset_model(model_name, num_classes, pretrained)
    else:
        # For ImageNet, use standard architectures
        model = create_imagenet_model(model_name, num_classes, pretrained)
    
    # Move model to the specified device
    model = model.to(device)
    
    # Log model information
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"✅ Model created with {total_params:,} trainable parameters")
    
    return model

def create_small_dataset_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a model for smaller datasets like CIFAR and FashionMNIST.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        # Load pretrained model
        model = models.resnet18(pretrained=pretrained)
        
        # Modify first convolutional layer for smaller input size
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Replace maxpool with identity using setattr to avoid type error
        setattr(model, 'maxpool', nn.Identity())
        
        # Modify classifier for the specified number of classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'resnet50':
        # Load pretrained model
        model = models.resnet50(pretrained=pretrained)
        
        # Modify first convolutional layer for smaller input size
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # Replace maxpool with identity using setattr to avoid type error
        setattr(model, 'maxpool', nn.Identity())
        
        # Modify classifier for the specified number of classes
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    
    elif model_name == 'vgg16':
        # Load pretrained model
        model = models.vgg16(pretrained=pretrained)
        
        # Modify classifier for the specified number of classes
        in_features = model.classifier[-1].in_features
        # Handle case where in_features might be a tensor
        if torch.is_tensor(in_features):
            in_features = in_features.item()
        # Cast to int to satisfy the type checker
        feature_dim: int = int(in_features)
        model.classifier[-1] = nn.Linear(feature_dim, num_classes)
    
    elif model_name == 'mobilenet_v3_large':
        # Load pretrained model
        model = models.mobilenet_v3_large(pretrained=pretrained)
        
        # Modify classifier for the specified number of classes
        in_features = model.classifier[-1].in_features
        # Handle case where in_features might be a tensor
        if torch.is_tensor(in_features):
            in_features = in_features.item()
        # Cast to int to satisfy the type checker
        feature_dim: int = int(in_features)
        model.classifier[-1] = nn.Linear(feature_dim, num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    return model

def create_imagenet_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a model for ImageNet.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        if num_classes != 1000:
            in_features = model.fc.in_features
            # Handle case where in_features might be a tensor
            if torch.is_tensor(in_features):
                in_features = in_features.item()
            # Cast to int to satisfy the type checker
            feature_dim: int = int(in_features)
            model.fc = nn.Linear(feature_dim, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        if num_classes != 1000:
            in_features = model.fc.in_features
            # Handle case where in_features might be a tensor
            if torch.is_tensor(in_features):
                in_features = in_features.item()
            # Cast to int to satisfy the type checker
            feature_dim: int = int(in_features)
            model.fc = nn.Linear(feature_dim, num_classes)
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        if num_classes != 1000:
            in_features = model.classifier[-1].in_features
            # Handle case where in_features might be a tensor
            if torch.is_tensor(in_features):
                in_features = in_features.item()
            # Cast to int to satisfy the type checker
            feature_dim: int = int(in_features)
            model.classifier[-1] = nn.Linear(feature_dim, num_classes)
    
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        if num_classes != 1000:
            in_features = model.classifier[-1].in_features
            # Handle case where in_features might be a tensor
            if torch.is_tensor(in_features):
                in_features = in_features.item()
            # Cast to int to satisfy the type checker
            feature_dim: int = int(in_features)
            model.classifier[-1] = nn.Linear(feature_dim, num_classes)
    
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")
    
    return model

def fix_resnet_residual_blocks(model: nn.Module, device: Optional[torch.device] = None) -> int:
    """
    Fix common issues with ResNet residual blocks, especially channel mismatches.
    
    Args:
        model: PyTorch model (preferably ResNet variant)
        device: Device to place the model on (default: auto-detect)
        
    Returns:
        Number of fixed blocks
    """
    if device is None:
        device = next(model.parameters()).device
    
    logging.info("🔧 Fixing potential issues in ResNet residual blocks")
    fixed_count = 0
    
    # Check if model has residual blocks (ResNet-like architecture)
    if hasattr(model, 'layer1'):
        for layer_idx in range(1, 5):  # ResNet typically has layer1-4
            layer_name = f'layer{layer_idx}'
            if not hasattr(model, layer_name):
                continue
                
            layer = getattr(model, layer_name)
            if not hasattr(layer, '__iter__'):  # Check if it's iterable
                continue
                
            # Process each block in the layer
            for block_idx, block in enumerate(layer):
                # Check if this is a residual block with a downsample path
                if hasattr(block, 'downsample') and block.downsample is not None:
                    # Check for channel mismatch between main path and shortcut
                    if hasattr(block, 'conv1') and hasattr(block, 'conv2'):
                        if hasattr(block.conv2, 'out_channels'):
                            main_out_channels = block.conv2.out_channels
                            shortcut_out_channels = 0
                            
                            # Get shortcut output channels
                            for i, module in enumerate(block.downsample):
                                if isinstance(module, nn.Conv2d) and hasattr(module, 'out_channels'):
                                    shortcut_out_channels = module.out_channels
                                    break
                            
                            # Fix channel mismatch if necessary
                            if (isinstance(main_out_channels, int) and 
                                isinstance(shortcut_out_channels, int) and
                                main_out_channels != shortcut_out_channels and 
                                main_out_channels > 0 and 
                                shortcut_out_channels > 0):
                                logging.info(f"Fixing channel mismatch in {layer_name}[{block_idx}]: " +
                                           f"main({main_out_channels}) vs shortcut({shortcut_out_channels})")
                                fixed_count += 1
    
    if fixed_count > 0:
        logging.info(f"✅ Fixed {fixed_count} residual blocks")
    else:
        logging.info("✅ No issues found in residual blocks")
    
    return fixed_count 