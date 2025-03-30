# Structured Pruning Project

This is a restructured, modular implementation of a structured channel pruning framework for deep neural networks. The project focuses on physical channel pruning to reduce model size and computational requirements while maintaining accuracy.

## Key Features

- **Multiple pruning methods**: Channel, Kernel, and Intra-kernel pruning
- **Accurate FLOPS calculation**: Precise measurement of computational requirements
- **Efficient model rebuilding**: Physical removal of pruned channels
- **Comprehensive visualization**: Training curves, parameter distribution, etc.
- **Modular design**: Each component can be used independently

## Project Structure

```
Restructured_SP/
├── config/
│   └── config.py - Configuration files and hyperparameters
├── core/
│   ├── pruning.py - Core pruning implementation
│   └── training.py - Training and fine-tuning utilities
├── data/
│   └── datasets.py - Dataset loaders
├── models/
│   └── model_factory.py - Model creation and modification
├── utils/
│   ├── logging_setup.py - Logging utilities
│   ├── metrics.py - Metrics calculation (FLOPS, parameters)
│   └── visualization.py - Result visualization
├── experiments/ - Experiment output directory
├── run.py - Main entry point
└── README.md - This file
```

## Installation and Requirements

This project requires PyTorch and other common deep learning libraries:

```bash
pip install torch torchvision matplotlib numpy
```

## Usage

### Running an Experiment

You can run a complete experiment with a single command:

```bash
python run.py --dataset cifar10 --model resnet18 --pruning-method channel --pruning-ratio 0.5
```

### Command Line Arguments

#### Basic Settings:

- `--dataset`: Dataset to use (cifar10, cifar100, imagenet, fashionmnist)
- `--model`: Model architecture (resnet18, resnet50, vgg16, mobilenet_v3_large)
- `--output-dir`: Directory to save experiment results

#### Training Settings:

- `--base-train-epochs`: Number of epochs for base training
- `--learning-rate`: Learning rate for training
- `--batch-size`: Batch size for training and evaluation
- `--weight-decay`: Weight decay for training
- `--patience`: Early stopping patience

#### Pruning Settings:

- `--pruning-method`: Method for pruning (channel, kernel, intra-kernel)
- `--pruning-ratio`: Pruning ratio (fraction of channels/parameters to remove)
- `--global-pruning`: Use global pruning across all layers

#### Fine-tuning Settings:

- `--finetune-epochs`: Number of epochs for fine-tuning after pruning
- `--finetune-lr`: Learning rate for fine-tuning

#### Execution Settings:

- `--no-cuda`: Disable CUDA even if available
- `--no-fp16`: Disable mixed precision training
- `--num-workers`: Number of worker threads for data loading
- `--resume`: Resume training from checkpoint if available
- `--skip-base-training`: Skip base training and start from a pretrained model
- `--skip-pruning`: Skip pruning step (for baseline evaluations)
- `--eval-only`: Only run evaluation on an existing model
- `--save-model`: Save models during training
- `--model-path`: Path to a saved model to load

## Example Workflows

### Train, Prune, and Fine-tune

```bash
python run.py --dataset cifar10 --model resnet18 --base-train-epochs 100 \
              --pruning-method channel --pruning-ratio 0.5 --finetune-epochs 20 \
              --save-model
```

### Use a Pretrained Model, Prune, and Fine-tune

```bash
python run.py --dataset cifar10 --model resnet18 --skip-base-training \
              --pruning-method kernel --pruning-ratio 0.5 --finetune-epochs 20
```

### Evaluate a Pruned Model

```bash
python run.py --dataset cifar10 --model resnet18 --eval-only \
              --model-path ./experiments/cifar10/resnet18_channel_cifar10_20230101_120000/resnet18_rebuilt.pth
```

## Pruning Methods

### Channel Pruning
Removes entire output channels from convolutional layers based on their L1-norm importance. This method is effective for reducing both model size and computation.

### Kernel Pruning
Removes entire convolutional kernels based on their L1-norm. This method targets specific filters within a convolutional layer.

### Intra-Kernel Pruning
Prunes connections within convolutional kernels, creating a sparse kernel pattern. This method maintains the overall network structure while reducing computation.

## Adding New Functionality

### Adding a New Pruning Method

1. Add the method to the StructuredPruning class in `core/pruning.py`
2. Add the method to the choices in the command line arguments in `run.py`

### Supporting a New Model Architecture

1. Add the model to the `create_small_dataset_model` or `create_imagenet_model` function in `models/model_factory.py`
2. Update the `MODEL_MAPPING` in `config/config.py` if needed

## Results and Metrics

After running an experiment, the following metrics are reported:

- Model parameters before and after pruning
- FLOPS before and after pruning
- Accuracy before and after pruning
- Training and fine-tuning curves
- Parameter distribution

Results are saved in the `experiments/` directory, organized by dataset and timestamp.

## Credits and References

This project is a restructured version of the original pruning framework, designed to be more modular and maintainable. It incorporates best practices for physical channel pruning as described in various research papers.

- [Network Slimming (SLIM)](https://arxiv.org/abs/1708.06519)
- [LAMP: Layer-Adaptive Magnitude-based Pruning](https://arxiv.org/abs/2010.07611)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
