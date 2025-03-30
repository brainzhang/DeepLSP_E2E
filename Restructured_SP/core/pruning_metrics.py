import os
import sys
import importlib
import importlib.util
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import logging
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

# pyright: reportCallIssue=false

def calculate_pruning_metrics(model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict[str, Any]:
    """
    Calculate metrics for a pruned model.
    
    Args:
        model: Pruned PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        
    Returns:
        Dictionary with metrics (original and pruned params, FLOPs, reduction percentages)
    """
    logging.info("📊 Calculating pruning metrics...")
    metrics = {}
    device = next(model.parameters()).device
    
    # Try to import FLOPs calculation libraries
    thop_available = False
    fvcore_available = False
    thop_module = None  # Initialize to avoid unbound errors
    
    try:
        import thop as thop_module
        thop_available = True
        logging.info("✅ Using thop for FLOPs calculation")
    except ImportError:
        logging.warning("⚠️ thop not found, trying alternative FLOPs calculators")
    
    if not thop_available:
        try:
            from fvcore.nn import FlopCountAnalysis
            fvcore_available = True
            logging.info("✅ Using fvcore for FLOPs calculation")
        except ImportError:
            logging.warning("⚠️ fvcore not found, will try custom FLOPs calculators")
    
    # Try custom metrics module as last resort
    metrics_module = None
    if not thop_available and not fvcore_available:
        try:
            import sys
            import importlib.util
            
            # Try to find metrics.py in utils directory or other common locations
            spec = importlib.util.find_spec("utils.metrics")
            if spec is None:
                for path in sys.path:
                    try:
                        metrics_path = os.path.join(path, "utils", "metrics.py")
                        if os.path.exists(metrics_path):
                            spec = importlib.util.spec_from_file_location("metrics", metrics_path)
                            break
                    except Exception:
                        continue
            
            if spec is not None and spec.loader is not None:
                metrics_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metrics_module)
                logging.info("✅ Using custom metrics module for calculations")
        except Exception as e:
            logging.error(f"❌ Error loading custom metrics module: {e}")
            metrics_module = None
    
    # Create a copy of the unpruned model by restoring original weights
    original_model = copy.deepcopy(model)
    
    # Check if model has pruning masks applied
    has_pruning_masks = False
    for module in original_model.modules():
        if hasattr(module, 'weight_orig') and hasattr(module, 'weight_mask'):
            has_pruning_masks = True
            # Restore original weights (remove mask effect)
            if isinstance(module.weight_orig, torch.Tensor):
                with torch.no_grad():
                    if hasattr(module, 'weight') and module.weight is not None:
                        module.weight.data.copy_(module.weight_orig.data)
            # Remove pruning
            prune.remove(module, 'weight')
    
    # 检测零权重剪枝 - 即使没有使用掩码，也可能通过将整个通道置零来进行剪枝
    if not has_pruning_masks:
        logging.warning("⚠️ No pruning masks detected. Checking for zero-weight channels...")
        
        # 计算每个卷积层中的零通道比例
        zero_channel_count = 0
        total_channel_count = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                weight = module.weight.data
                out_channels = weight.shape[0]
                
                # 检查每个输出通道是否全为零
                for i in range(out_channels):
                    channel_weights = weight[i]
                    if torch.sum(torch.abs(channel_weights)) == 0:
                        zero_channel_count += 1
                
                total_channel_count += out_channels
        
        # 如果有显著数量的零通道，则认为已应用了通道剪枝
        zero_channel_ratio = zero_channel_count / max(1, total_channel_count)
        if zero_channel_ratio > 0.01:  # 超过1%的通道为零
            has_pruning_masks = True  # 将零通道也视为一种剪枝
            logging.info(f"📊 检测到零通道剪枝: {zero_channel_count}/{total_channel_count} 通道 ({zero_channel_ratio:.2%})")
        else:
            logging.warning("⚠️ 未检测到显著的剪枝效果，模型可能未被剪枝")
    
    if not has_pruning_masks:
        logging.warning("⚠️ 无法检测到任何剪枝标记或零权重通道。使用原始参数进行度量计算。")
    
    # 确保计算实际的非零参数数量
    pruned_params_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    original_params = sum(p.numel() for p in original_model.parameters() if p.requires_grad)
    
    # 计算剪枝模型中的非零参数
    nonzero_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            nonzero_params += torch.count_nonzero(param).item()
    
    metrics['original_params'] = original_params
    metrics['pruned_params'] = nonzero_params
    metrics['param_reduction'] = 100.0 * (1.0 - nonzero_params / original_params) if original_params > 0 else 0
    
    # Calculate FLOPs using available methods
    if thop_available and thop_module is not None:
        # Use thop for FLOPs calculation
        try:
            # Create correctly sized input tensors
            dummy_input = torch.randn(input_size).to(device)
            
            # Calculate FLOPs for original model
            original_model.eval()
            temp = thop_module.profile(original_model, inputs=(dummy_input,), verbose=False)
            # thop sometimes returns 2 or 3 values, handle both cases
            if isinstance(temp, tuple) and len(temp) >= 2:
                macs = temp[0]
                original_flops = macs * 2  # Convert MACs to FLOPs
            
                # Calculate FLOPs for pruned model
                model.eval() 
                temp = thop_module.profile(model, inputs=(dummy_input,), verbose=False)
                macs = temp[0]
                pruned_flops = macs * 2
                
                metrics['original_flops'] = original_flops
                metrics['pruned_flops'] = pruned_flops
                metrics['flops_reduction'] = 100.0 * (1.0 - pruned_flops / original_flops) if original_flops > 0 else 0
                
                logging.info(f"📊 Original model: Parameters {original_params:,}, FLOPs {original_flops/1e6:.2f}M")
                logging.info(f"📊 Pruned model: Parameters {nonzero_params:,}, FLOPs {pruned_flops/1e6:.2f}M")
                logging.info(f"📊 Reduction: Parameters {metrics['param_reduction']:.2f}%, FLOPs {metrics['flops_reduction']:.2f}%")
            else:
                logging.warning("⚠️ Unexpected output format from thop.profile")
                thop_available = False
        except Exception as e:
            logging.error(f"❌ Error calculating FLOPs with thop: {e}")
            thop_available = False
            
    elif fvcore_available:
        # Use fvcore for FLOPs calculation
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.randn(input_size).to(device)
            
            # Calculate FLOPs for original model
            original_model.eval()
            flops_analysis = FlopCountAnalysis(original_model, dummy_input)
            original_flops = flops_analysis.total()
            
            # Calculate FLOPs for pruned model
            model.eval()
            flops_analysis = FlopCountAnalysis(model, dummy_input)
            pruned_flops = flops_analysis.total()
            
            metrics['original_flops'] = original_flops
            metrics['pruned_flops'] = pruned_flops
            metrics['flops_reduction'] = 100.0 * (1.0 - pruned_flops / original_flops) if original_flops > 0 else 0
            
            logging.info(f"📊 Original model: Parameters {original_params:,}, FLOPs {original_flops/1e6:.2f}M")
            logging.info(f"📊 Pruned model: Parameters {nonzero_params:,}, FLOPs {pruned_flops/1e6:.2f}M")
            logging.info(f"📊 Reduction: Parameters {metrics['param_reduction']:.2f}%, FLOPs {metrics['flops_reduction']:.2f}%")
        except Exception as e:
            logging.error(f"❌ Error calculating FLOPs with fvcore: {e}")
            fvcore_available = False
            
    elif metrics_module is not None and hasattr(metrics_module, 'calculate_flops'):
        # Use custom metrics module
        try:
            original_flops = metrics_module.calculate_flops(original_model, input_size)
            pruned_flops = metrics_module.calculate_flops(model, input_size)
            
            metrics['original_flops'] = original_flops
            metrics['pruned_flops'] = pruned_flops
            metrics['flops_reduction'] = 100.0 * (1.0 - pruned_flops / original_flops) if original_flops > 0 else 0
            
            logging.info(f"📊 Original model: Parameters {original_params:,}, FLOPs {original_flops/1e6:.2f}M")
            logging.info(f"📊 Pruned model: Parameters {nonzero_params:,}, FLOPs {pruned_flops/1e6:.2f}M")
            logging.info(f"📊 Reduction: Parameters {metrics['param_reduction']:.2f}%, FLOPs {metrics['flops_reduction']:.2f}%")
        except Exception as e:
            logging.error(f"❌ Error calculating FLOPs with custom metrics: {e}")
    
    # 如果所有FLOP计算方法都失败，则自行计算
    if 'flops_reduction' not in metrics:
        # 手动计算FLOP
        logging.info("📊 使用手动方法计算FLOPS...")
        
        # 初始化计数器
        original_flops_count = 0
        pruned_flops_count = 0
        
        # 遍历并计算每层的FLOPS
        for (name_orig, module_orig), (name_pruned, module_pruned) in zip(
            original_model.named_modules(), model.named_modules()):
            
            # 只处理计算密集型层
            if isinstance(module_orig, nn.Conv2d) and isinstance(module_pruned, nn.Conv2d):
                # 获取参数
                k_h, k_w = module_orig.kernel_size if isinstance(module_orig.kernel_size, tuple) else (module_orig.kernel_size, module_orig.kernel_size)
                in_c_orig = module_orig.in_channels
                out_c_orig = module_orig.out_channels
                in_c_pruned = module_pruned.in_channels
                out_c_pruned = module_pruned.out_channels
                groups_orig = module_orig.groups
                groups_pruned = module_pruned.groups
                
                # 计算特征图尺寸 (假设输入是input_size指定的大小)
                stride = module_orig.stride[0] if isinstance(module_orig.stride, tuple) else module_orig.stride
                padding = module_orig.padding[0] if isinstance(module_orig.padding, tuple) else module_orig.padding
                dilation = module_orig.dilation[0] if isinstance(module_orig.dilation, tuple) else module_orig.dilation
                
                # 计算输出特征图尺寸
                h_in, w_in = input_size[2], input_size[3]
                h_out = (h_in + 2 * int(padding) - int(dilation) * (k_h - 1) - 1) // int(stride) + 1
                w_out = (w_in + 2 * int(padding) - int(dilation) * (k_w - 1) - 1) // int(stride) + 1
                
                # 计算原始FLOPS: 2 * H_out * W_out * K_h * K_w * C_in * C_out / groups
                original_layer_flops = 2 * h_out * w_out * k_h * k_w * in_c_orig * out_c_orig // groups_orig
                original_flops_count += original_layer_flops
                
                # 如果结构真的变了 (通道数减少)
                if in_c_pruned < in_c_orig or out_c_pruned < out_c_orig:
                    pruned_layer_flops = 2 * h_out * w_out * k_h * k_w * in_c_pruned * out_c_pruned // groups_pruned
                    pruned_flops_count += pruned_layer_flops
                    logging.info(f"📊 层{name_orig}: 通道从{in_c_orig}x{out_c_orig}减少到{in_c_pruned}x{out_c_pruned}, FLOPS减少: {100*(1-pruned_layer_flops/original_layer_flops):.2f}%")
                else:
                    # 结构没变，但可能有很多零权重
                    nonzero_weights = torch.count_nonzero(module_pruned.weight).item()
                    total_weights = module_pruned.weight.numel()
                    nonzero_ratio = nonzero_weights / total_weights if total_weights > 0 else 1.0
                    
                    pruned_layer_flops = original_layer_flops * nonzero_ratio
                    pruned_flops_count += pruned_layer_flops
                    logging.info(f"📊 层{name_orig}: 权重稀疏度: {1-nonzero_ratio:.2f}, FLOPS减少: {100*(1-nonzero_ratio):.2f}%")
            
            elif isinstance(module_orig, nn.Linear) and isinstance(module_pruned, nn.Linear):
                # 线性层FLOPS: 2 * in_features * out_features
                original_layer_flops = 2 * module_orig.in_features * module_orig.out_features
                original_flops_count += original_layer_flops
                
                if module_pruned.in_features < module_orig.in_features or module_pruned.out_features < module_orig.out_features:
                    pruned_layer_flops = 2 * module_pruned.in_features * module_pruned.out_features
                    pruned_flops_count += pruned_layer_flops
                else:
                    nonzero_weights = torch.count_nonzero(module_pruned.weight).item()
                    total_weights = module_pruned.weight.numel()
                    nonzero_ratio = nonzero_weights / total_weights if total_weights > 0 else 1.0
                    
                    pruned_layer_flops = original_layer_flops * nonzero_ratio
                    pruned_flops_count += pruned_layer_flops
        
        # 更新度量结果
        if original_flops_count > 0:
            metrics['original_flops'] = original_flops_count
            metrics['pruned_flops'] = pruned_flops_count
            flops_reduction = 100.0 * (1.0 - pruned_flops_count / original_flops_count)
            metrics['flops_reduction'] = flops_reduction
            
            logging.info(f"📊 手动计算 - 原始FLOPS: {original_flops_count/1e6:.2f}M")
            logging.info(f"📊 手动计算 - 剪枝后FLOPS: {pruned_flops_count/1e6:.2f}M")
            logging.info(f"📊 手动计算 - FLOPS减少: {flops_reduction:.2f}%")
        else:
            # 最后的办法，使用参数减少比例作为估计
            param_reduction_ratio = metrics['param_reduction'] / 100.0
            metrics['flops_reduction'] = metrics['param_reduction']  # 或者更保守地基于参数稀疏度估计
            
            logging.warning("⚠️ 无法直接计算FLOPS，使用参数减少比例作为估计值")
            logging.info(f"📊 估计FLOPS减少: {metrics['flops_reduction']:.2f}%")
    
    return metrics

def get_model_pruning_ratio(model: nn.Module) -> float:
    """
    Calculate the overall pruning ratio of a model.
    
    Args:
        model: PyTorch model with pruning applied
        
    Returns:
        Pruning ratio (0.0 - 1.0)
    """
    total_weights = 0
    zero_weights = 0
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight = module.weight.data
            total_weights += weight.numel()
            zero_weights += (weight == 0).sum().item()
    
    return zero_weights / total_weights if total_weights > 0 else 0.0 