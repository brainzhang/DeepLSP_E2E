import os
import json
import logging
from pathlib import Path

# Model configuration
MODEL_MAPPING = {
    "imagenet": {
        "default": "resnet50",
        "models": ["resnet18", "resnet50", "vgg16", "mobilenet_v3_large"]
    },
    "cifar10": {
        "default": "resnet18",
        "models": ["resnet18", "resnet50", "vgg16", "mobilenet_v3_large"]
    },
    "cifar100": {
        "default": "resnet18",
        "models": ["resnet18", "resnet50", "vgg16", "mobilenet_v3_large"]
    },
    "fashionmnist": {
        "default": "resnet18",
        "models": ["resnet18", "resnet50", "vgg16", "mobilenet_v3_large"]
    },
}

# Default parameters
DEFAULT_CONFIG = {
    "pruning_methods": ["l1norm", "random", "lamp", "slim", "group_norm", "structured_pruning"],
    "default_pruning_ratio": 0.5,
    "default_finetune_epochs": 10,
    "default_finetune_lr": 0.001,
    "use_fp16": True,
}

# Input sizes for different datasets
INPUT_SIZES = {
    "imagenet": (1, 3, 224, 224),
    "cifar10": (1, 3, 32, 32),
    "cifar100": (1, 3, 32, 32),
    "fashionmnist": (1, 1, 28, 28),
}

def get_input_size(dataset):
    """Get the input size for a specific dataset"""
    return INPUT_SIZES.get(dataset, (1, 3, 32, 32))

def get_experiment_dir(dataset, model_name, pruning_method=None, timestamp=None):
    """Build experiment directory path"""
    experiment_dir = Path("/mnt/d/Ubuntu/phd/Prune_Methods/Restructured_SP/experiments") / dataset
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    if pruning_method and timestamp:
        # For multiple pruning methods, join them with "_"
        if isinstance(pruning_method, list):
            pruning_method = "_".join(pruning_method)
        return experiment_dir / f"{model_name}_{pruning_method}_{dataset}_{timestamp}"
    return experiment_dir

def save_config(config, save_path):
    """Save configuration to JSON file"""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f"✅ Configuration saved to: {save_path}")

def load_config(config_path):
    """Load configuration from JSON file"""
    if not os.path.exists(config_path):
        logging.warning(f"⚠️ Configuration file not found: {config_path}")
        return DEFAULT_CONFIG
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"✅ Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"❌ Failed to load configuration: {e}")
        return DEFAULT_CONFIG 