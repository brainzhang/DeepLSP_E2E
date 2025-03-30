import logging
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def save_results(results, save_path):
    """
    Save experiment results to a JSON file
    
    Args:
        results: Dictionary of results
        save_path: Path to save the JSON file
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save results to JSON
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        logging.info(f"✅ Results saved to: {save_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save results: {e}")

def plot_training_history(history, output_path):
    """
    Plot the training history.
    
    Args:
        history: Dict with training history
        output_path: Path to save the plot
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['loss'], label='Train')
        if 'val_loss' in history:
            ax1.plot(history['val_loss'], label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot accuracy
        # 兼容旧版和新版history键名
        acc_key = 'accuracy' if 'accuracy' in history else 'acc'
        val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
        
        ax2.plot(history[acc_key], label='Train')
        if val_acc_key in history:
            ax2.plot(history[val_acc_key], label='Validation')
        ax2.set_title('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logging.info(f"📊 Training history saved to: {output_path}")
    except Exception as e:
        logging.error(f"❌ Failed to plot training history: {e}")

def plot_pruning_comparison(results, save_path=None):
    """
    Plot pruning comparison (accuracy vs. FLOPs reduction)
    
    Args:
        results: Dictionary containing pruning results for different methods
        save_path: Path to save the plot (optional)
    """
    try:
        plt.figure(figsize=(10, 6))
        
        # Extract data for plotting
        methods = []
        accuracies = []
        flops_reductions = []
        param_reductions = []
        
        for method, data in results.items():
            if 'accuracy' in data and 'flops_reduction' in data:
                methods.append(method)
                accuracies.append(data['accuracy'])
                flops_reductions.append(data['flops_reduction'])
                if 'param_reduction' in data:
                    param_reductions.append(data['param_reduction'])
                else:
                    param_reductions.append(0)
        
        # Plot accuracy vs. FLOPs reduction
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        x = range(len(methods))
        ax1.bar(x, accuracies, width=0.4, align='edge', label='Accuracy', color='blue', alpha=0.7)
        ax1.set_ylabel('Accuracy (%)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim((0, 100))
        
        ax2 = ax1.twinx()
        ax2.bar([i+0.4 for i in x], flops_reductions, width=0.4, align='edge', label='FLOPs Reduction', color='red', alpha=0.7)
        ax2.set_ylabel('FLOPs Reduction (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim((0, 100))
        
        plt.title('Pruning Methods Comparison')
        plt.xticks([i+0.2 for i in x], methods, rotation=45)
        
        # Create a separate legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            logging.info(f"✅ Pruning comparison plot saved to: {save_path}")
        
        plt.close()
    except Exception as e:
        logging.error(f"❌ Failed to plot pruning comparison: {e}")

def plot_parameter_distribution(model, layer_types=None, save_path=None):
    """
    Plot parameter distribution for different layers in the model
    
    Args:
        model: PyTorch model
        layer_types: List of layer types to include (e.g., ['Conv2d', 'Linear'])
        save_path: Path to save the plot (optional)
    """
    try:
        import torch.nn as nn
        
        # Define layer types to include if not specified
        if layer_types is None:
            layer_types = [nn.Conv2d, nn.Linear]
        
        # Collect parameter statistics for each layer
        layer_data = []
        
        for name, module in model.named_modules():
            if any(isinstance(module, layer_type) for layer_type in layer_types):
                # Get parameter count
                param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                # Get parameter statistics (mean and std)
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_mean = module.weight.data.mean().item()
                    weight_std = module.weight.data.std().item()
                    
                    layer_data.append({
                        'name': name,
                        'type': module.__class__.__name__,
                        'params': param_count,
                        'weight_mean': weight_mean,
                        'weight_std': weight_std
                    })
        
        if not layer_data:
            logging.warning("⚠️ No layers found for parameter distribution plot")
            return
        
        # Sort layers by parameter count
        layer_data.sort(key=lambda x: x['params'], reverse=True)
        
        # Plot parameter distribution
        plt.figure(figsize=(12, 8))
        
        # Plot parameter count
        plt.subplot(2, 1, 1)
        names = [f"{d['name']} ({d['type']})" for d in layer_data]
        params = [d['params'] for d in layer_data]
        
        plt.bar(range(len(names)), params)
        plt.xticks(range(len(names)), [f"{i+1}" for i in range(len(names))], rotation=0)
        plt.title('Parameter Count by Layer')
        plt.xlabel('Layer Index')
        plt.ylabel('Parameter Count')
        plt.yscale('log')
        
        # Add a legend mapping indices to layer names
        legend_text = '\n'.join([f"{i+1}: {name}" for i, name in enumerate(names)])
        plt.figtext(1.02, 0.5, legend_text, fontsize=8, verticalalignment='center')
        
        # Plot weight statistics
        plt.subplot(2, 1, 2)
        means = [d['weight_mean'] for d in layer_data]
        stds = [d['weight_std'] for d in layer_data]
        
        plt.errorbar(range(len(names)), means, yerr=stds, fmt='o', capsize=5)
        plt.xticks(range(len(names)), [f"{i+1}" for i in range(len(names))], rotation=0)
        plt.title('Weight Statistics by Layer')
        plt.xlabel('Layer Index')
        plt.ylabel('Weight Mean ± Std')
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            logging.info(f"✅ Parameter distribution plot saved to: {save_path}")
        
        plt.close()
    except Exception as e:
        logging.error(f"❌ Failed to plot parameter distribution: {e}") 