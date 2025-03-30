#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportCallIssue=false, reportArgumentType=false

import os
import sys
import argparse
import logging
import torch
import json
from datetime import datetime
from pathlib import Path
import torch.onnx
import torch.nn as nn
import copy
import numpy as np
import time
from typing import Tuple, Dict, Any

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import modules from the project
from config.config import MODEL_MAPPING, get_input_size, get_experiment_dir
from utils.logging_setup import setup_logging, log_system_info
from data.datasets import get_dataloaders
from models.model_factory import create_model, fix_resnet_residual_blocks

# Import specific functions from modules
from core.training import train_model, fine_tune_model, evaluate_model
from core.pruning import StructuredPruning
from core.pruning_metrics import calculate_pruning_metrics
from utils.metrics import count_parameters, calculate_flops, measure_inference_time, optimize_model_for_inference
from utils.visualization import plot_training_history, save_results

# 这些可能需要单独安装
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("⚠️ ONNXRuntime not available. Install with: pip install onnxruntime-gpu")

try:
    import torch_pruning as tp
    from torch_pruning import MetaPruner
    TORCH_PRUNING_AVAILABLE = True
except ImportError:
    TORCH_PRUNING_AVAILABLE = False
    logging.warning("⚠️ torch_pruning not available. Install with: pip install torch-pruning")

# 尝试导入TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logging.warning("⚠️ TensorRT not available. Follow https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html to install.")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run structured pruning experiments')
    
    # Basic experiment settings

    parser.add_argument('model', type=str, default='resnet18',
                        help='Model architecture to use (defaults to dataset default)')
    parser.add_argument('dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'fashionmnist'],
                        help='Dataset to use for experiments')
    parser.add_argument('--output-dir', type=str, default='/mnt/d/Ubuntu/phd/Prune_Methods/Restructured_SP/experiments',
                        help='Directory to save experiment results')
    
    # Training settings
    parser.add_argument('--base-train-epochs', type=int, default=100,
                        help='Number of epochs for base training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and evaluation')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay for training')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    
    # Pruning settings
    parser.add_argument('--pruning-method', nargs='+', 
                        default=['channel', 'kernel', 'intra-kernel'],
                        choices=['channel', 'kernel', 'intra-kernel'],
                        help='Methods for pruning (can specify multiple methods)')
    parser.add_argument('--pruning-ratio', type=float, default=0.5,
                        help='Pruning ratio (fraction of channels to remove)')
    parser.add_argument('--global-pruning', action='store_true',
                        help='Use global pruning across all layers (default: layer-wise)')
    
    # Fine-tuning settings
    parser.add_argument('--finetune-epochs', type=int, default=10,
                        help='Number of epochs for fine-tuning after pruning')
    parser.add_argument('--finetune-lr', type=float, default=0.001,
                        help='Learning rate for fine-tuning')
    
    # Execution settings
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA even if available')
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint if available')
    parser.add_argument('--skip-base-training', action='store_true',
                        help='Skip base training and start from a pretrained model')
    parser.add_argument('--skip-pruning', action='store_true',
                        help='Skip pruning step (for baseline evaluations)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on an existing model')
    
    # 推理优化选项
    parser.add_argument('--use-tensorrt', action='store_true',
                        help='Use TensorRT for optimized inference')
    parser.add_argument('--use-onnx', action='store_true',
                        help='Use ONNX Runtime for optimized inference')
    parser.add_argument('--align-channels', action='store_true',
                        help='Align pruned channels to hardware-friendly values (e.g., multiples of 32)')
    parser.add_argument('--hardware-align', type=int, default=32,
                        help='Channel alignment value for hardware-friendly pruning (default: 32)')
    parser.add_argument('--skip-rebuild', action='store_true',
                        help='Skip rebuilding the model structure after pruning (keep zero weights)')
    
    # Model loading/saving
    parser.add_argument('--save-model', action='store_true',
                        help='Save models during training')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to a saved model to load')
    
    return parser.parse_args()

def run_experiment(args):
    """
    Run a complete pruning experiment based on provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Set up timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Select model architecture based on dataset if not specified
    if args.model is None:
        model_name = MODEL_MAPPING[args.dataset]['default']
    else:
        model_name = args.model
    
    # Create experiment directory and setup logging
    experiment_dir = get_experiment_dir(args.dataset, model_name, args.pruning_method, timestamp)
    log_path = setup_logging(args.dataset, model_name, timestamp)
    
    # Log experiment configuration
    logging.info(f"🧪 Starting experiment with {model_name} on {args.dataset}")
    logging.info(f"🧪 Pruning methods: {', '.join(args.pruning_method)}, ratio: {args.pruning_ratio}")
    
    # 初始化关键变量，避免未绑定错误
    pruned_model = None
    
    # Get device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device, cuda_available = log_system_info()
    
    if not use_cuda:
        device = torch.device('cpu')
        logging.info("Using CPU for computation (CUDA disabled or not available)")
    
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if cuda_available:
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Load dataset
    logging.info(f"📚 Loading {args.dataset} dataset")
    data = get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    train_loader = data['train_loader']
    test_loader = data['test_loader']
    num_classes = data['num_classes']
    input_size = data['input_size']
    
    # Create results dictionary to store experiment metrics
    results = {
        'dataset': args.dataset,
        'model': model_name,
        'pruning_method': args.pruning_method,
        'pruning_ratio': args.pruning_ratio,
        'global_pruning': args.global_pruning,
        'timestamp': timestamp,
    }
    
    # Create and train the model (if not evaluating only)
    if not args.eval_only:
        if args.model_path is not None and os.path.exists(args.model_path):
            # Load existing model
            logging.info(f"⬇️ Loading model from {args.model_path}")
            
            # Create model architecture
            model = create_model(
                model_name=model_name,
                dataset=args.dataset,
                num_classes=num_classes,
                pretrained=False,
                device=device
            )
            
            # Load saved weights
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            # Fix any potential issues in residual blocks
            fix_resnet_residual_blocks(model, device)
            
            logging.info(f"✅ Model loaded successfully: {model_name}")
        elif args.skip_base_training:
            # Create model with pretrained weights
            logging.info(f"⬇️ Creating model with pretrained weights: {model_name}")
            
            model = create_model(
                model_name=model_name,
                dataset=args.dataset,
                num_classes=num_classes,
                pretrained=True,
                device=device
            )
            
            # Fix any potential issues in residual blocks
            fix_resnet_residual_blocks(model, device)
            
            # Perform a quick evaluation before pruning
            orig_loss, orig_acc = evaluate_model(model, test_loader)
            logging.info(f"📊 Pretrained model - Loss: {orig_loss:.4f}, Accuracy: {orig_acc:.2f}%")
            results['pretrained_accuracy'] = orig_acc
        else:
            # Create and train model from scratch
            logging.info(f"🏗️ Creating and training model from scratch: {model_name}")
            
            model = create_model(
                model_name=model_name,
                dataset=args.dataset,
                num_classes=num_classes,
                pretrained=False,
                device=device
            )
            
            # Fix any potential issues in residual blocks
            fix_resnet_residual_blocks(model, device)
            
            # Train the model
            use_fp16 = not args.no_fp16
            train_results = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=test_loader,
                num_epochs=args.base_train_epochs,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                patience=args.patience,
                fp16=use_fp16
            )
            
            # Save training history
            results['base_training'] = {
                'epochs': len(train_results['history']['loss']),
                'best_epoch': train_results['best_epoch'],
                'best_val_acc': train_results['best_val_acc'],
                'training_time': train_results['training_time']
            }
            
            # Plot training history
            plot_path = os.path.join(experiment_dir, f"{model_name}_training_history.png")
            plot_training_history(train_results['history'], plot_path)
            
            # Save the trained model if requested
            if args.save_model:
                model_save_path = os.path.join(experiment_dir, f"{model_name}_base_trained.pth")
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"💾 Base trained model saved to: {model_save_path}")
        
        # Calculate model size and FLOPs before pruning
        orig_params = count_parameters(model)
        orig_flops = calculate_flops(model, input_size)
        
        # 创建模型副本用于推理测量，保留原始模型
        logging.info("📊 准备测量原始模型推理时间...")
        orig_model_copy = copy.deepcopy(model)
        
        # 优化模型以提高推理速度
        logging.info("🚀 优化原始模型以提高推理速度...")
        optimized_original = optimize_model_for_inference(orig_model_copy, input_size=input_size)
        
        # 测量推理时间
        orig_inference_time = measure_inference_time(optimized_original, input_size=input_size, use_jit=True)
        
        results['original_model'] = {
            'parameters': orig_params,
            'flops': orig_flops,
            'inference_time_ms': orig_inference_time
        }
        
        logging.info(f"📊 Original model - Parameters: {orig_params:,}, FLOPs: {orig_flops/1e6:.2f}M, Inference time: {orig_inference_time:.2f}ms")
        
        # Skip pruning if requested
        if not args.skip_pruning:
            # Apply pruning methods
            # 使用标准剪枝方法
            logging.info(f"✂️ Applying pruning methods: {', '.join(args.pruning_method)}, ratio: {args.pruning_ratio}")
            
            pruner = StructuredPruning(model, pruning_ratio=args.pruning_ratio)
            pruned_model = model
            
            # 应用剪枝（此时只是置零权重）
            for method in args.pruning_method:
                logging.info(f"✂️ Applying {method} pruning method")
                pruned_model = pruner.prune_model(method=method, global_pruning=args.global_pruning)
            
            # 如果torch_pruning可用，使用专业工具进行结构化剪枝
            if not args.skip_rebuild and TORCH_PRUNING_AVAILABLE:
                logging.info("🔄 使用torch_pruning专业工具进行结构化剪枝...")
                try:
                    # 使用专业剪枝工具 (全局函数)
                    pruned_model = prune_model_properly(pruned_model, args.pruning_ratio, args.dataset)
                except Exception as e:
                    logging.error(f"❌ 专业剪枝工具调用失败: {e}")
                    logging.warning("⚠️ 回退到基本剪枝方法")
            
            # 评估置零权重后的模型
            masked_loss, masked_acc = evaluate_model(pruned_model, test_loader)
            logging.info(f"📊 Pruned model (with masks) - Loss: {masked_loss:.4f}, Accuracy: {masked_acc:.2f}%")
            
            results['masked_model'] = {
                'accuracy': masked_acc,
                'loss': masked_loss
            }
            
            # 更新模型引用
            model = pruned_model
            
            # Fine-tune the pruned model if requested
            if args.finetune_epochs > 0:
                logging.info(f"🔄 Fine-tuning pruned model for {args.finetune_epochs} epochs")
                
                # Fine-tune with mixed precision if enabled
                use_fp16 = not args.no_fp16
                finetune_results = fine_tune_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=test_loader,
                    num_epochs=args.finetune_epochs,
                    learning_rate=args.finetune_lr,
                    fp16=use_fp16
                )
                
                # Save fine-tuning history
                results['fine_tuning'] = {
                    'epochs': len(finetune_results['history']['loss']),
                    'best_epoch': finetune_results['best_epoch'],
                    'best_val_acc': finetune_results['best_val_acc'],
                    'training_time': finetune_results['training_time']
                }
                
                # Plot fine-tuning history
                plot_path = os.path.join(experiment_dir, f"{model_name}_finetuning_history.png")
                plot_training_history(finetune_results['history'], plot_path)
                
                # Save the fine-tuned model if requested
                if args.save_model:
                    model_save_path = os.path.join(experiment_dir, f"{model_name}_finetuned.pth")
                    torch.save(model.state_dict(), model_save_path)
                    logging.info(f"💾 Fine-tuned model saved to: {model_save_path}")
            
            # Rebuild the model to remove pruned channels
            logging.info("🔨 Calculating pruning metrics")
            
            # Get metrics without rebuilding the model
            metrics = calculate_pruning_metrics(
                model=model,
                input_size=input_size
            )
            
            # Keep the same model instead of rebuilding
            rebuilt_model = model
            
            # Update model to rebuilt version
            model = rebuilt_model
            
            # Validate the rebuilt model with a single forward pass
            try:
                model.eval()
                with torch.no_grad():
                    dummy_input = torch.randn(1, input_size[1], input_size[2], input_size[3], device=device)
                    _ = model(dummy_input)
                logging.info("✅ Validated rebuilt model with test forward pass")
                
                # Measure inference time after pruning
                logging.info("📊 Measuring inference time after pruning...")
                
                # 使用当前模型作为评估基准
                model_for_eval = model  # 保存用于评估的标准模型
                
                # 如果启用了通道对齐
                if args.align_channels:
                    logging.info(f"🔧 对齐模型通道到{args.hardware_align}的倍数...")
                    model = align_pruning_channels(model, align_to=args.hardware_align)
                
                # 优化模型以提高推理速度
                logging.info("🚀 优化模型以提高推理速度...")
                
                # 创建ONNX模型路径
                onnx_path = os.path.join(experiment_dir, f"{model_name}_pruned.onnx")
                
                # 确保实验目录存在
                os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
                
                # 测量推理时间
                pruned_inference_time = 0.0
                
                # 根据用户选择的优化方法测量推理时间
                if args.use_tensorrt and TENSORRT_AVAILABLE:
                    logging.info("🚀 使用TensorRT进行推理优化...")
                    
                    # 创建TensorRT引擎
                    engine_path = os.path.join(experiment_dir, f"{model_name}_pruned.engine")
                    engine = create_tensorrt_engine(
                        model, 
                        input_size=input_size, 
                        onnx_path=onnx_path, 
                        engine_path=engine_path
                    )
                    
                    # 使用TensorRT测量推理时间
                    if engine is not None:
                        pruned_inference_time = measure_tensorrt_inference_time(
                            engine,
                            input_size=input_size
                        )
                    else:
                        logging.warning("⚠️ TensorRT引擎创建失败，回退到PyTorch推理")
                        optimized_model = optimize_model_for_inference(copy.deepcopy(model), input_size=input_size)
                        pruned_inference_time = measure_inference_time(optimized_model, input_size=input_size, use_jit=True)
                
                elif args.use_onnx and ONNX_AVAILABLE:
                    logging.info("🚀 使用ONNX Runtime进行推理优化...")
                    
                    # 导出为ONNX格式
                    dummy_input = torch.randn(input_size, device=device)
                    torch.onnx.export(
                        model,
                        dummy_input,
                        onnx_path,
                        export_params=True,
                        opset_version=13,
                        do_constant_folding=True,
                        input_names=['input'],
                        output_names=['output'],
                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                    )
                    
                    # 使用ONNX Runtime测量推理时间
                    pruned_inference_time = measure_onnx_inference_time(
                        onnx_path,
                        input_size=input_size,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )
                    
                else:
                    # 使用PyTorch JIT优化
                    optimized_model = optimize_model_for_inference(copy.deepcopy(model), input_size=input_size)
                    pruned_inference_time = measure_inference_time(optimized_model, input_size=input_size, use_jit=True)
                
                metrics['inference_time_ms'] = pruned_inference_time
                
                # Update inference time speedup
                if 'original_params' in metrics and metrics.get('inference_time_ms', 0) > 0:
                    orig_time = results['original_model']['inference_time_ms']
                    time_speedup = 100.0 * (1.0 - pruned_inference_time / orig_time) if orig_time > 0 else 0
                    metrics['inference_speedup'] = time_speedup
                    
                    # Add inference time reduction metric (negative when time increases)
                    time_reduction = 100.0 * (orig_time - pruned_inference_time) / orig_time if orig_time > 0 else 0
                    metrics['inference_time_reduction'] = time_reduction
                    
                    logging.info(f"📊 推理时间变化: {orig_time:.2f}ms → {pruned_inference_time:.2f}ms")
                    logging.info(f"📊 推理时间加速率: {time_speedup:.2f}%, 时间减少率: {time_reduction:.2f}%")
                
            except Exception as e:
                logging.error(f"❌ Error validating rebuilt model: {e}")
                # Revert to original model
                logging.warning("⚠️ Reverting to original model due to rebuilding error")
                model = pruned_model
            
            # Save metrics from pruning
            results['pruning'] = metrics
            
            # 如果实际执行了结构重建，重新计算FLOPs
            if not args.skip_rebuild:
                # 重新计算参数和FLOPs
                new_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                param_reduction = 100.0 * (1.0 - new_params / orig_params) if orig_params > 0 else 0
                
                # 重新计算FLOPs
                new_flops = calculate_flops(model, input_size)
                flops_reduction = 100.0 * (1.0 - new_flops / orig_flops) if orig_flops > 0 else 0
                
                # 更新指标
                metrics['pruned_params'] = new_params
                metrics['param_reduction'] = param_reduction
                metrics['pruned_flops'] = new_flops
                metrics['flops_reduction'] = flops_reduction
                
                logging.info(f"📊 重建后 - 参数: {new_params:,} ({param_reduction:.2f}% 减少)")
                logging.info(f"📊 重建后 - FLOPs: {new_flops/1e6:.2f}M ({flops_reduction:.2f}% 减少)")
            
            # Evaluate final model - 直接使用原始模型进行评估
            final_loss, final_acc = evaluate_model(model, test_loader)
            logging.info(f"📊 Final rebuilt model - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%")
            
            # Use metrics from the pruned model for final model parameters
            results['final_model'] = {
                'accuracy': final_acc,
                'loss': final_loss,
                'parameters': metrics.get('pruned_params', orig_params),
                'flops': metrics.get('pruned_flops', orig_flops),
                'inference_time_ms': metrics.get('inference_time_ms', 0),
                'param_reduction': metrics.get('param_reduction', 0.0),
                'flops_reduction': metrics.get('flops_reduction', 0.0),
                'inference_speedup': metrics.get('inference_speedup', 0.0),
                'inference_time_reduction': metrics.get('inference_time_reduction', 0.0)
            }
            
    else:
        # Evaluation only mode
        if args.model_path is None:
            logging.error("❌ Model path must be provided for evaluation-only mode")
            return None, None
        
        if not os.path.exists(args.model_path):
            logging.error(f"❌ Model file not found: {args.model_path}")
            return None, None
        
        # Create model architecture
        model = create_model(
            model_name=model_name,
            dataset=args.dataset,
            num_classes=num_classes,
            pretrained=False,
            device=device
        )
        
        # Load saved weights
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Fix any potential issues in residual blocks
        fix_resnet_residual_blocks(model, device)
        
        logging.info(f"✅ Model loaded successfully: {model_name}")
    
    # Final evaluation
    logging.info("📊 执行最终评估...")
    final_loss, final_acc = evaluate_model(model, test_loader)
    logging.info(f"📊 Final evaluation - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%")
    
    # Do NOT recalculate parameters and FLOPs here - use values from pruning metrics
    if 'final_model' not in results:
        results['final_model'] = {}
    
    # Update only accuracy and loss, keeping pruning metrics intact
    results['final_model'].update({
        'accuracy': final_acc,
        'loss': final_loss
    })
    
    # If we're in evaluation-only mode or skipped pruning, we need to calculate metrics
    if args.eval_only or args.skip_pruning:
        final_params = count_parameters(model)
        final_flops = calculate_flops(model, input_size)
        
        results['final_model'].update({
            'parameters': final_params,
            'flops': final_flops
        })
        
        # Calculate reduction ratios if original model is available
        if 'original_model' in results:
            orig_params = results['original_model']['parameters']
            orig_flops = results['original_model']['flops']
            
            if orig_params > 0:
                param_reduction = 100.0 * (1.0 - final_params / orig_params)
                results['final_model']['param_reduction'] = param_reduction
                
                if orig_flops > 0:
                    flops_reduction = 100.0 * (1.0 - final_flops / orig_flops)
                    results['final_model']['flops_reduction'] = flops_reduction
                    
                    logging.info(f"📊 Model size reduction: {param_reduction:.2f}%, FLOPs reduction: {flops_reduction:.2f}%")
    # For pruned models, use the pruning metrics directly (already stored in results['final_model'] earlier)
    else:
        logging.info(f"📊 Model size reduction: {results['final_model']['param_reduction']:.2f}%, " 
                    f"FLOPs reduction: {results['final_model']['flops_reduction']:.2f}%")
    
    # Save results
    results_path = os.path.join(experiment_dir, f"{model_name}_results.json")
    save_results(results, results_path)
    logging.info(f"💾 Experiment results saved to: {results_path}")
    
    # Create a summary
    logging.info(f"\n{'='*60}\n📝 EXPERIMENT SUMMARY\n{'='*60}")
    logging.info(f"Model: {model_name}, Dataset: {args.dataset}")
    logging.info(f"Pruning methods: {', '.join(args.pruning_method)}, Ratio: {args.pruning_ratio}")
    
    if 'original_model' in results:
        logging.info(f"Original model - Parameters: {results['original_model']['parameters']:,}, "
                    f"FLOPs: {results['original_model']['flops']/1e6:.2f}M, "
                    f"Inference time: {results['original_model'].get('inference_time_ms', 0):.2f}ms")
    
    if 'final_model' in results:
        logging.info(f"Final model - Parameters: {results['final_model']['parameters']:,}, "
                    f"FLOPs: {results['final_model']['flops']/1e6:.2f}M, "
                    f"Inference time: {results['final_model'].get('inference_time_ms', 0):.2f}ms")
        
        if 'param_reduction' in results['final_model']:
            logging.info(f"Parameter reduction: {results['final_model']['param_reduction']:.2f}%")
        
        if 'flops_reduction' in results['final_model']:
            logging.info(f"FLOPs reduction: {results['final_model']['flops_reduction']:.2f}%")
        
        if 'inference_speedup' in results['final_model']:
            logging.info(f"Inference speedup: {results['final_model']['inference_speedup']:.2f}%")
        
        if 'inference_time_reduction' in results['final_model']:
            logging.info(f"Inference time reduction: {results['final_model']['inference_time_reduction']:.2f}%")
        
        logging.info(f"Final accuracy: {results['final_model']['accuracy']:.2f}%")
    
    logging.info(f"{'='*60}")
    
    return results, model

def create_tensorrt_engine(model, input_size, onnx_path=None, engine_path=None, precision='fp16'):
    """
    将PyTorch模型转换为TensorRT引擎以加速推理
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸 (batch_size, channels, height, width)
        onnx_path: ONNX模型路径，如果不提供则创建临时文件
        engine_path: TensorRT引擎保存路径
        precision: 精度模式 ('fp32', 'fp16', 'int8')
        
    Returns:
        TensorRT引擎
    """
    if not TENSORRT_AVAILABLE:
        logging.error("❌ TensorRT not available for optimized inference")
        return None
    
    # 确保模型处于评估模式
    model.eval()
    
    # 如果未提供ONNX路径，创建临时文件
    if onnx_path is None:
        import tempfile
        temp_dir = tempfile.gettempdir()
        onnx_path = os.path.join(temp_dir, f"temp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx")
    
    # 如果未提供引擎路径，根据ONNX路径创建
    if engine_path is None:
        engine_path = onnx_path.replace(".onnx", ".engine")
    
    # 创建随机输入
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size, device=device)
    
    # 先导出为ONNX格式
    logging.info(f"📦 导出模型为ONNX格式: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # 创建TensorRT引擎
    logging.info(f"🚀 构建TensorRT引擎: {engine_path}")
    
    try:
        # 确保TensorRT已成功导入
        import tensorrt as trt  # pyright: ignore[reportAttributeAccessIssue]
            
        # 创建TensorRT builder和网络
        logger = trt.Logger(trt.Logger.WARNING)  # pyright: ignore[reportAttributeAccessIssue]
        builder = trt.Builder(logger)  # pyright: ignore[reportAttributeAccessIssue]
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  # pyright: ignore[reportAttributeAccessIssue]
        parser = trt.OnnxParser(network, logger)  # pyright: ignore[reportAttributeAccessIssue]
        
        # 解析ONNX文件
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                logging.error("❌ 解析ONNX文件失败")
                for error in range(parser.num_errors):
                    logging.error(f"Error {error}: {parser.get_error(error)}")
                return None
        
        # 配置TensorRT
        config = builder.create_builder_config()
        # 新版TensorRT API使用set_memory_pool_limit
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # pyright: ignore[reportAttributeAccessIssue]
        
        # 为动态批量大小创建优化配置文件
        profile = builder.create_optimization_profile()
        # 设置批量大小范围: 最小、最优、最大
        min_batch = 1
        opt_batch = input_size[0]  # 优化大小为输入批量大小
        max_batch = input_size[0] * 2  # 最大批量大小设为输入的2倍
        
        # 配置'input'张量的优化参数
        profile.set_shape(
            "input",  # 输入名称
            (min_batch, input_size[1], input_size[2], input_size[3]),  # 最小尺寸
            (opt_batch, input_size[1], input_size[2], input_size[3]),  # 最优尺寸
            (max_batch, input_size[1], input_size[2], input_size[3])   # 最大尺寸
        )
        config.add_optimization_profile(profile)
        
        # 设置精度
        if precision == 'fp16' and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)  # pyright: ignore[reportAttributeAccessIssue]
            logging.info("⚙️ 启用FP16精度")
        elif precision == 'int8' and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)  # pyright: ignore[reportAttributeAccessIssue]
            logging.info("⚙️ 启用INT8精度")
        
        # 构建引擎
        serialized_engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        # 创建运行时和引擎
        runtime = trt.Runtime(logger)  # pyright: ignore[reportAttributeAccessIssue]
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
            
        logging.info("✅ TensorRT引擎创建成功")
        return engine
    
    except ImportError as e:
        logging.error(f"❌ TensorRT导入失败: {e}")
        return None
    except Exception as e:
        logging.error(f"❌ 创建TensorRT引擎失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return None

def measure_tensorrt_inference_time(engine, input_size, num_iterations=100):
    """
    使用TensorRT引擎测量推理时间
    
    Args:
        engine: TensorRT引擎
        input_size: 输入尺寸 (batch_size, channels, height, width)
        num_iterations: 测量迭代次数
        
    Returns:
        float: 平均推理时间（毫秒）
    """
    if engine is None:
        logging.error("❌ 无效的TensorRT引擎")
        return 0
    
    # 检查必要的库是否已导入
    if not TENSORRT_AVAILABLE:
        logging.error("❌ TensorRT未正确导入")
        return 0
        
    try:
        # 避免静态分析器的警告，动态导入所需模块
        import importlib
        cuda_module = importlib.import_module("pycuda.driver")
        
        # 创建CUDA上下文
        context = engine.create_execution_context()
        
        # 准备输入和输出内存 - 直接使用numpy创建数组，避免类型问题
        h_input = np.zeros(input_size, dtype=np.float32)
        for i in range(input_size[0]):
            for c in range(input_size[1]):
                for h in range(input_size[2]):
                    for w in range(input_size[3]):
                        h_input[i,c,h,w] = np.random.random()
        
        # 获取输出形状 - 使用新版TensorRT API
        # 旧版: output_shape = engine.get_binding_shape(1)
        # 使用context或engine获取绑定维度的替代方法
        binding_idx = 1  # 输出绑定索引
        if hasattr(engine, 'get_binding_dimensions'):
            # 尝试使用get_binding_dimensions
            output_dims = engine.get_binding_dimensions(binding_idx)
            output_shape = (input_size[0], output_dims[1])  # 假设为(N, C)格式
        elif hasattr(context, 'get_binding_shape'):
            # 尝试从上下文获取
            output_shape = context.get_binding_shape(binding_idx)
        else:
            # 最后的备选方案 - 假设输出与模型对应
            logging.warning("⚠️ 无法获取输出形状，使用默认值(N, 1000)")
            output_shape = (input_size[0], 1000)  # 假设为1000类分类
        
        logging.info(f"📊 TensorRT输出形状: {output_shape}")
        h_output = np.zeros((input_size[0], output_shape[1]), dtype=np.float32)
        
        # 使用getattr动态获取PyCUDA函数，避免静态分析器警告
        try:
            # 分配GPU内存
            mem_alloc_fn = getattr(cuda_module, "mem_alloc")
            d_input = mem_alloc_fn(h_input.nbytes)
            d_output = mem_alloc_fn(h_output.nbytes)
            
            # 创建CUDA流
            stream_cls = getattr(cuda_module, "Stream")
            stream = stream_cls()
            
            # 内存拷贝函数
            htod_async_fn = getattr(cuda_module, "memcpy_htod_async")
            dtoh_async_fn = getattr(cuda_module, "memcpy_dtoh_async")
            
            # 预热
            for _ in range(10):
                htod_async_fn(d_input, h_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                dtoh_async_fn(h_output, d_output, stream)
                stream.synchronize()
            
            # 测量时间
            start_time = time.time()
            for _ in range(num_iterations):
                htod_async_fn(d_input, h_input, stream)
                context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
                dtoh_async_fn(h_output, d_output, stream)
                stream.synchronize()
        except (AttributeError, ImportError) as e:
            # 如果找不到特定方法或导入失败，尝试使用同步版本
            logging.warning(f"⚠️ 使用同步推理接口: {e}")
            
            # 简化的同步推理版本
            import pycuda.driver
            
            # 使用同步版本的接口，添加type:ignore注释忽略静态分析错误
            d_input = pycuda.driver.mem_alloc(h_input.nbytes)  # type: ignore
            d_output = pycuda.driver.mem_alloc(h_output.nbytes)  # type: ignore
            
            # 预热
            for _ in range(10):
                pycuda.driver.memcpy_htod(d_input, h_input)  # type: ignore
                context.execute_v2(bindings=[int(d_input), int(d_output)])
                pycuda.driver.memcpy_dtoh(h_output, d_output)  # type: ignore
            
            # 测量时间
            start_time = time.time()
            for _ in range(num_iterations):
                pycuda.driver.memcpy_htod(d_input, h_input)  # type: ignore
                context.execute_v2(bindings=[int(d_input), int(d_output)])
                pycuda.driver.memcpy_dtoh(h_output, d_output)  # type: ignore
        
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / num_iterations) * 1000  # 转换为毫秒
        
        logging.info(f"📊 TensorRT平均推理时间: {avg_time_ms:.2f} ms ({num_iterations} 次迭代)")
        
        # 释放资源
        del context
        
        return avg_time_ms
    except ImportError as e:
        logging.error(f"❌ PyCUDA导入失败: {e}")
        return 0
    except Exception as e:
        logging.error(f"❌ TensorRT推理测量失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 0

def measure_onnx_inference_time(onnx_path, input_size, num_iterations=100, device='cuda'):
    """
    使用ONNX Runtime测量推理时间
    
    Args:
        onnx_path: ONNX模型路径
        input_size: 输入尺寸 (batch_size, channels, height, width)
        num_iterations: 测量迭代次数
        device: 运行设备 ('cuda' 或 'cpu')
        
    Returns:
        float: 平均推理时间（毫秒）
    """
    if not ONNX_AVAILABLE:
        logging.error("❌ ONNX Runtime not available for optimized inference")
        return 0
    
    if not os.path.exists(onnx_path):
        logging.error(f"❌ ONNX模型不存在: {onnx_path}")
        return 0
    
    # 创建ONNX会话
    logging.info(f"📦 加载ONNX模型: {onnx_path}")
    try:
        import onnxruntime as ort
            
        if device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
            
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # 准备输入 - 直接创建numpy数组
        input_name = session.get_inputs()[0].name
        dummy_input = np.zeros(input_size, dtype=np.float32)
        for i in range(input_size[0]):
            for c in range(input_size[1]):
                for h in range(input_size[2]):
                    for w in range(input_size[3]):
                        dummy_input[i,c,h,w] = np.random.random()
        
        # 预热
        for _ in range(10):
            _ = session.run(None, {input_name: dummy_input})
        
        # 测量时间
        start_time = time.time()
        for _ in range(num_iterations):
            _ = session.run(None, {input_name: dummy_input})
        
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / num_iterations) * 1000  # 转换为毫秒
        
        logging.info(f"📊 ONNX Runtime平均推理时间: {avg_time_ms:.2f} ms ({num_iterations} 次迭代)")
        
        return avg_time_ms
    
    except ImportError as e:
        logging.error(f"❌ ONNX Runtime导入失败: {e}")
        return 0
    except Exception as e:
        logging.error(f"❌ ONNX Runtime推理失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 0

# 添加新的通道对齐剪枝功能
def round_to_multiple(number, multiple):
    """将数字四舍五入到最接近的倍数"""
    return multiple * round(number / multiple)

def align_pruning_channels(model, align_to=32):
    """
    调整模型各层的输出通道数，使其对齐到指定的倍数
    这可以大幅提高GPU推理性能
    
    Args:
        model: PyTorch模型
        align_to: 对齐的通道数倍数
        
    Returns:
        调整后的模型
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 获取当前通道数
            out_channels = module.out_channels
            # 向下对齐到最近的倍数
            aligned_channels = (out_channels // align_to) * align_to
            # 如果对齐会减少太多通道，则向上对齐
            if out_channels - aligned_channels > align_to / 2:
                aligned_channels += align_to
            
            # 如果需要调整且不会使通道数变为0
            if aligned_channels != out_channels and aligned_channels > 0:
                logging.info(f"⚙️ 对齐层 {name} 的输出通道: {out_channels} -> {aligned_channels}")
                
                # 创建新的卷积层
                aligned_conv = nn.Conv2d(
                    module.in_channels,
                    aligned_channels,
                    module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=module.bias is not None,
                    padding_mode=module.padding_mode
                )
                
                # 复制原始权重
                with torch.no_grad():
                    if aligned_channels > out_channels:
                        # 扩展通道
                        aligned_conv.weight[:out_channels] = module.weight
                        # 处理bias，避免对None使用下标
                        if module.bias is not None and aligned_conv.bias is not None:
                            aligned_conv.bias[:out_channels] = module.bias
                    else:
                        # 收缩通道
                        aligned_conv.weight = nn.Parameter(module.weight[:aligned_channels])
                        # 处理bias，避免对None使用下标
                        if module.bias is not None and aligned_conv.bias is not None:
                            aligned_conv.bias = nn.Parameter(module.bias[:aligned_channels])
                
                # 替换模型中的层
                # 注意：这种直接替换可能会导致问题，特别是在复杂网络中
                # 在生产环境中应该使用更完善的层替换方法
                setattr(model, name.split('.')[-1], aligned_conv)
    
    return model

def rebuild_pruned_model(model):
    """
    创建结构化的剪枝模型，真正移除零权重通道，而不仅仅是将它们置为零。
    实现方式是找出非零通道，创建新的更小的网络。
    同时处理后续的BatchNorm层，确保维度匹配。
    
    Args:
        model: 已经应用了权重掩码的模型
        
    Returns:
        new_model: 结构更小的新模型
    """
    logging.info("⚙️ 正在重建模型，移除零权重通道...")
    
    # 获取原始模型的设备
    device = next(model.parameters()).device
    
    # 检查网络类型，如果是ResNet，给出警告
    if 'resnet' in str(type(model)).lower():
        logging.warning("⚠️ 检测到ResNet模型。ResNet中的残差连接要求输入/输出通道数匹配。")
        logging.warning("⚠️ 本实现可能不适用于ResNet。建议跳过重建步骤或使用专业的剪枝库。")
        logging.warning("⚠️ 继续执行，但可能会出现层间通道不匹配的错误。")
    
    # 创建新模型实例
    new_model = copy.deepcopy(model)
    
    # 这是一个简化版实现，适用于简单网络
    # 对于复杂网络，应该使用像torch_pruning这样的库来保持依赖关系一致
    
    # 首先获取所有Conv2d层的零通道掩码
    zero_masks = {}
    for name, module in new_model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data
            out_channels = weight.shape[0]
            
            # 找出零通道
            nonzero_mask = []
            for i in range(out_channels):
                channel_norm = torch.sum(torch.abs(weight[i]))
                nonzero_mask.append(channel_norm > 0)
            
            nonzero_mask = torch.tensor(nonzero_mask, dtype=torch.bool, device=device)
            nonzero_count = torch.sum(nonzero_mask).item()
            
            # 如果有通道可以移除
            if nonzero_count < out_channels and nonzero_count > 0:
                zero_masks[name] = (nonzero_mask, nonzero_count)
    
    # 创建卷积层名称到其后的BatchNorm层的映射
    conv_to_bn = {}
    prev_name = None
    
    # 通过模型的命名模块找出卷积层与其后的BatchNorm层
    for name, module in new_model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and prev_name is not None and prev_name in zero_masks:
            conv_to_bn[prev_name] = name
        
        if isinstance(module, nn.Conv2d):
            prev_name = name
        else:
            prev_name = None
    
    # 遍历并修改卷积层和相应的BatchNorm层
    for conv_name, (nonzero_mask, nonzero_count) in zero_masks.items():
        # 获取卷积层
        conv_path = conv_name.split('.')
        conv_module = new_model
        for part in conv_path:
            conv_module = getattr(conv_module, part)
        
        # 对于ResNet，我们只处理某些安全的层，跳过可能破坏结构的层
        if 'resnet' in str(type(model)).lower():
            # 跳过下采样层(downsample)和可能影响残差连接的层
            if 'downsample' in conv_name or conv_name.endswith('conv1') or conv_name.endswith('conv3'):
                logging.warning(f"⚠️ 跳过ResNet关键层 {conv_name} 以保持网络结构")
                continue
        
        logging.info(f"🔍 层 {conv_name}: 发现 {conv_module.out_channels - nonzero_count}/{conv_module.out_channels} 个零通道")
        
        # 创建新的卷积层，输出通道减少
        try:
            # 获取参数
            in_channels = conv_module.in_channels
            out_channels = nonzero_count
            kernel_size = conv_module.kernel_size
            stride = conv_module.stride
            padding = conv_module.padding
            dilation = conv_module.dilation
            groups = conv_module.groups  # 保留原始分组
            bias = conv_module.bias is not None
            
            # 创建新的卷积层并移至相同设备
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups if groups==1 else max(1, out_channels//in_channels*in_channels),
                bias=bias
            ).to(device)  # 移至与原模型相同的设备
            
            # 复制非零通道的权重
            idx = 0
            for i in range(conv_module.out_channels):
                if nonzero_mask[i]:
                    new_conv.weight.data[idx] = conv_module.weight.data[i]
                    # 额外检查确保bias存在
                    if bias and conv_module.bias is not None and new_conv.bias is not None:
                        new_conv.bias.data[idx] = conv_module.bias.data[i]
                    idx += 1
            
            # 替换卷积层
            parent_name = '.'.join(conv_path[:-1])
            attr_name = conv_path[-1]
            
            if parent_name:
                parent = new_model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, attr_name, new_conv)
            else:
                setattr(new_model, attr_name, new_conv)
            
            logging.info(f"✅ 成功替换卷积层 {conv_name}，输出通道从 {conv_module.out_channels} 减少到 {nonzero_count}")
            
            # 检查是否有相应的BatchNorm层需要更新
            if conv_name in conv_to_bn:
                bn_name = conv_to_bn[conv_name]
                bn_path = bn_name.split('.')
                bn_module = new_model
                for part in bn_path:
                    bn_module = getattr(bn_module, part)
                
                # 创建新的BatchNorm层
                new_bn = nn.BatchNorm2d(
                    num_features=nonzero_count,
                    eps=bn_module.eps,
                    momentum=bn_module.momentum,
                    affine=bn_module.affine,
                    track_running_stats=bn_module.track_running_stats
                ).to(device)
                
                # 复制非零通道的参数
                if bn_module.affine:
                    idx = 0
                    for i in range(bn_module.num_features):
                        if nonzero_mask[i]:
                            new_bn.weight.data[idx] = bn_module.weight.data[i]
                            new_bn.bias.data[idx] = bn_module.bias.data[i]
                            idx += 1
                
                # 复制运行时统计数据
                if bn_module.track_running_stats:
                    idx = 0
                    for i in range(bn_module.num_features):
                        if nonzero_mask[i]:
                            # 确保不访问None
                            if hasattr(bn_module, 'running_mean') and bn_module.running_mean is not None and \
                               hasattr(new_bn, 'running_mean') and new_bn.running_mean is not None:
                                new_bn.running_mean[idx] = bn_module.running_mean[i]
                            
                            if hasattr(bn_module, 'running_var') and bn_module.running_var is not None and \
                               hasattr(new_bn, 'running_var') and new_bn.running_var is not None:
                                new_bn.running_var[idx] = bn_module.running_var[i]
                            
                            idx += 1
                
                # 替换BatchNorm层
                parent_name = '.'.join(bn_path[:-1])
                attr_name = bn_path[-1]
                
                if parent_name:
                    parent = new_model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, attr_name, new_bn)
                else:
                    setattr(new_model, attr_name, new_bn)
                
                logging.info(f"✅ 成功替换BatchNorm层 {bn_name}，特征数从 {bn_module.num_features} 减少到 {nonzero_count}")
                
        except Exception as e:
            logging.warning(f"⚠️ 处理层 {conv_name} 时出错: {e}")
    
    # 计算参数数量以验证结构是否真的变小
    orig_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    new_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    reduction = (orig_params - new_params) / orig_params * 100 if orig_params > 0 else 0
    
    logging.info(f"📊 参数: {orig_params:,} → {new_params:,} (减少了 {reduction:.2f}%)")
    
    # 确保整个模型都在正确的设备上
    new_model = new_model.to(device)
    
    # 尝试运行一次前向传播以验证模型结构
    try:
        dummy_input = torch.randn(1, 3, 32, 32).to(device)  # 示例输入，根据模型调整尺寸
        new_model.eval()
        with torch.no_grad():
            _ = new_model(dummy_input)
        logging.info("✅ 重建后的模型验证成功，结构一致")
    except Exception as e:
        logging.error(f"❌ 模型结构不一致，可能需要专业的结构化剪枝工具: {e}")
        # 由于我们不希望中断执行流程，这里返回原始模型
        logging.warning("⚠️ 返回原始模型（带零权重）")
        return model
    
    return new_model

def prune_model_properly(model, pruning_ratio=0.5, dataset='cifar10'):
    try:
        import torch_pruning as tp
        
        logging.info("🔪 开始结构化剪枝，使用torch_pruning库...")
        
        # 获取设备信息
        device = next(model.parameters()).device
        
        # 创建示例输入
        if dataset == 'imagenet':
            example_inputs = torch.randn(1, 3, 224, 224).to(device)
        else:
            example_inputs = torch.randn(1, 3, 32, 32).to(device)
            
        # 创建剪枝器 - 使用2.x版本API
        importance = tp.importance.MagnitudeImportance(p=1)  # L1范数
        
        # 保护分类层
        ignored_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                ignored_layers.append(module)
                logging.info(f"🛡️ 保护分类层 {name}")
                
        # 创建剪枝器
        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs, 
            importance,
            ch_sparsity=pruning_ratio,  # 通道稀疏度
            ignored_layers=ignored_layers,  # 保护层
        )
        
        # 执行剪枝
        pruner.step()
        
        # 验证模型是否正常工作
        model.eval()
        with torch.no_grad():
            output = model(example_inputs)
            logging.info(f"✅ 剪枝后模型输出形状: {output.shape}")
            
        # 计算参数数量
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"📊 剪枝后的参数数量: {param_count:,}")
        
        return model
        
    except Exception as e:
        logging.error(f"❌ 结构化剪枝失败: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return model

if __name__ == "__main__":
    args = parse_args()
    results, model = run_experiment(args) 