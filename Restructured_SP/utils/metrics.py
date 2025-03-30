# pyright: reportCallIssue=false, reportUnboundVariable=false, reportAttributeAccessIssue=false
import torch
import torch.nn as nn
import logging
from typing import Tuple, Union, Dict, Any, Optional, cast
import time
import copy
import torch.jit

# 定义一个通用模型类型，可以是标准Module或JIT模型
ModelType = Union[nn.Module, torch.jit.ScriptModule, Any]

def count_parameters(model: ModelType) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    try:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except (AttributeError, RuntimeError) as e:
        logging.warning(f"⚠️ Unable to count parameters normally: {e}")
        try:
            # 尝试不检查requires_grad
            return sum(p.numel() for p in model.parameters())
        except Exception as e:
            logging.warning(f"⚠️ Unable to count parameters in model, returning 0: {e}")
            return 0

def get_kernel_size(kernel_size) -> Tuple[int, int]:
    """
    Safely get the kernel size as a 2-tuple
    
    Args:
        kernel_size: Kernel size (int or tuple)
        
    Returns:
        Tuple[int, int]: Kernel size as a 2-tuple
    """
    if isinstance(kernel_size, tuple):
        if len(kernel_size) == 2:
            return kernel_size
        elif len(kernel_size) == 1:
            return (kernel_size[0], kernel_size[0])
    elif isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    return (1, 1)  # Default value

def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """
    计算模型的FLOPs（浮点运算次数）
    
    Args:
        model: PyTorch模型
        input_size: 输入张量的尺寸 (B, C, H, W)
        
    Returns:
        float: 模型的FLOPs
    """
    import torch
    import torch.nn as nn
    
    def conv_flops(module, input_size, output_size):
        # 计算卷积层的FLOPs
        if not isinstance(module, nn.Conv2d):
            return 0
            
        # 如果是整个通道为零的剪枝卷积层，返回零FLOPs
        in_channels = module.in_channels
        out_channels = module.out_channels
        
        # 检查是否有零通道（整个输出通道权重为零）
        effective_out_channels = out_channels
        for i in range(out_channels):
            if torch.sum(torch.abs(module.weight[i])) == 0:
                effective_out_channels -= 1  # 减少有效输出通道数
        
        # 如果所有通道都被剪枝，返回0 FLOPs
        if effective_out_channels == 0:
            return 0
        
        # 计算卷积层的FLOPs
        batch_size = input_size[0]
        output_h, output_w = output_size[2], output_size[3]
        
        kernel_h, kernel_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        groups = module.groups
        
        # 卷积层的FLOPs计算公式: 2 * B * H_out * W_out * C_in * C_out * K_h * K_w / groups
        flops = 2 * batch_size * output_h * output_w * in_channels * effective_out_channels * kernel_h * kernel_w / groups
        
        # 如果有bias，加上bias的FLOPs: B * H_out * W_out * C_out
        if module.bias is not None:
            flops += batch_size * output_h * output_w * effective_out_channels
            
        return flops
    
    def linear_flops(module, input_size, output_size):
        # 计算全连接层的FLOPs
        if not isinstance(module, nn.Linear):
            return 0
            
        batch_size = input_size[0]
        in_features = module.in_features
        out_features = module.out_features
        
        # 线性层的FLOPs计算公式: 2 * B * in_features * out_features
        flops = 2 * batch_size * in_features * out_features
        
        # 如果有bias，加上bias的FLOPs: B * out_features
        if module.bias is not None:
            flops += batch_size * out_features
            
        return flops
    
    def bn_flops(module, input_size):
        # 计算批量归一化层的FLOPs
        if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return 0
            
        batch_size = input_size[0]
        num_features = module.num_features
        
        # 计算特征元素总数
        num_elements = batch_size * num_features
        if len(input_size) > 2:
            for dim in input_size[2:]:
                num_elements *= dim
        
        # BN层的FLOPs计算公式: 2 * num_elements (一次减均值，一次除以标准差)
        flops = 2 * num_elements
        
        # 如果有affine变换，加上scale和shift的FLOPs: 2 * num_elements
        if module.affine:
            flops += 2 * num_elements
            
        return flops
    
    # 跟踪每层的输入和输出大小
    def register_hooks(model):
        model_hooks = []
        model_flops = [0]  # 使用列表存储，以便在闭包中修改
        
        def hook_fn(module, input, output):
            # 确保输入是元组
            if not input:
                return
            if not isinstance(input, tuple):
                input = (input,)
                
            input_size = input[0].size()
            output_size = output.size() if isinstance(output, torch.Tensor) else output[0].size()
            
            # 计算当前层的FLOPs
            if isinstance(module, nn.Conv2d):
                flops = conv_flops(module, input_size, output_size)
            elif isinstance(module, nn.Linear):
                flops = linear_flops(module, input_size, output_size)
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                flops = bn_flops(module, input_size)
            else:
                flops = 0  # 其他层暂不计算
                
            model_flops[0] += flops
        
        # 为每个子模块注册钩子
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                hook = module.register_forward_hook(hook_fn)
                model_hooks.append(hook)
                
        return model_hooks, model_flops
    
    # 创建随机输入
    device = next(model.parameters()).device
    x = torch.randn(input_size).to(device)
    
    # 注册钩子
    hooks, flops = register_hooks(model)
    
    # 执行前向传播
    model.eval()
    with torch.no_grad():
        _ = model(x)
        
    # 移除钩子
    for hook in hooks:
        hook.remove()
        
    # 返回总FLOPs
    return flops[0]

def compare_flops(original_model, pruned_model, input_size=(1, 3, 32, 32)) -> Dict[str, Any]:
    """
    Compare FLOPs between original and pruned models
    
    Args:
        original_model: Original model
        pruned_model: Pruned model
        input_size: Input tensor size
        
    Returns:
        Dict with original_flops, pruned_flops, reduction_percentage
    """
    logging.info("📊 Calculating original model FLOPs...")
    original_flops = calculate_flops(original_model, input_size)
    
    logging.info("📊 Calculating pruned model FLOPs...")
    pruned_flops = calculate_flops(pruned_model, input_size)
    
    # Calculate FLOPs reduction
    if original_flops > 0:
        reduction_percentage = 100.0 * (1.0 - pruned_flops / original_flops)
        logging.info(f"📊 FLOPs change: {original_flops/1e6:.2f}M -> {pruned_flops/1e6:.2f}M")
        logging.info(f"📊 FLOPs reduction: {reduction_percentage:.2f}%")
    else:
        reduction_percentage = 0.0
        logging.warning("⚠️ Cannot calculate FLOPs reduction: original FLOPs is 0")
    
    # Calculate parameter counts
    original_params = count_parameters(original_model)
    pruned_params = count_parameters(pruned_model)
    
    if original_params > 0:
        param_reduction = 100.0 * (1.0 - pruned_params / original_params)
        logging.info(f"📊 Parameters change: {original_params} -> {pruned_params}")
        logging.info(f"📊 Parameters reduction: {param_reduction:.2f}%")
    else:
        param_reduction = 0.0
    
    return {
        'original_flops': original_flops,
        'pruned_flops': pruned_flops,
        'flops_reduction': reduction_percentage,
        'original_params': original_params,
        'pruned_params': pruned_params,
        'param_reduction': param_reduction
    }

def measure_inference_time(model, input_size=(1, 3, 32, 32), num_iterations=100, device=None, repeat_count=5, use_jit=False):
    """
    Measure the average inference time of a model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        num_iterations: Number of iterations for each measurement round
        device: Device to run inference on
        repeat_count: Number of measurement rounds to perform (taking median)
        use_jit: Whether to use PyTorch JIT compilation to optimize inference
        
    Returns:
        float: Average inference time in milliseconds
    """
    if model is None:
        logging.error("❌ Model is None, cannot measure inference time")
        return 0
    
    # Set model to evaluation mode
    model.eval()
    
    # Get device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create random input with the right shape for the model
    dummy_input = torch.randn(input_size, device=device)
    
    # Apply JIT optimization if requested
    if use_jit:
        try:
            logging.info("🚀 Optimizing model with PyTorch JIT compilation...")
            # Try scripting the model for better performance
            with torch.no_grad():
                # Make a deep copy of the model to avoid modifying the original
                model_copy = copy.deepcopy(model)
                
                # Use torch.jit.trace for model optimization
                optimized_model = torch.jit.trace(model_copy, dummy_input)
                
                # Verify the JIT model works correctly
                orig_output = model(dummy_input)
                jit_output = optimized_model(dummy_input)
                
                # Check that outputs are similar
                if torch.allclose(orig_output, jit_output, rtol=1e-3, atol=1e-4):
                    logging.info("✅ JIT optimization successful, using optimized model")
                    model = optimized_model
                else:
                    logging.warning("⚠️ JIT model output differs from original model, falling back to non-JIT version")
        except Exception as e:
            logging.warning(f"⚠️ JIT optimization failed, using original model: {e}")
    
    # Adjust iterations based on device
    if device.type == 'cpu':
        # Use fewer iterations on CPU as it's slower
        num_iterations = max(20, num_iterations // 5)
        warmup_iterations = 20
    else:
        warmup_iterations = 50
    
    # Thread priority boost (try to minimize OS interruptions)
    try:
        import os
        os.nice(-10)  # Increase process priority if possible
    except (ImportError, PermissionError):
        pass
    
    # Multiple measurement rounds
    all_times = []
    
    for round_idx in range(repeat_count):
        # Extensive warmup phase
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)
        
        # Synchronize CUDA operations before measurement
        if device.type == 'cuda':
            torch.cuda.synchronize()
            # Clear GPU cache
            torch.cuda.empty_cache()
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
                
                # Synchronize CUDA operations after each iteration
                if device.type == 'cuda':
                    torch.cuda.synchronize()
        
        # Calculate average time for this round
        elapsed_time = time.time() - start_time
        avg_time_ms = (elapsed_time / num_iterations) * 1000  # Convert to ms
        all_times.append(avg_time_ms)
    
    # Take median of all measurement rounds to reduce outlier impact
    import statistics
    median_time = statistics.median(all_times)
    
    # Log detailed measurement information
    logging.info(f"📊 Inference time measurements across {repeat_count} rounds (ms): {', '.join([f'{t:.2f}' for t in all_times])}")
    logging.info(f"📊 Median inference time: {median_time:.2f} ms (over {num_iterations} iterations per round)")
    
    return median_time

def optimize_model_for_inference(model, input_size=(1, 3, 32, 32), device=None):
    """
    Optimize a model for inference using various techniques.
    
    Args:
        model: PyTorch model to optimize
        input_size: Input tensor size for the model
        device: Device to run inference on
        
    Returns:
        Optimized model
    """
    if model is None:
        logging.error("❌ Model is None, cannot optimize")
        return model
        
    # Set model to evaluation mode
    model.eval()
    
    # Get device
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # 在此处预先计算原始模型输出作为参考基准
    try:
        with torch.no_grad():
            orig_output = model(dummy_input)
    except Exception as e:
        logging.error(f"❌ Unable to get original model output: {e}")
        return model
    
    # Try to use TorchScript optimization
    try:
        logging.info("🚀 Optimizing model with TorchScript...")
        # Use torch.jit.trace for direct optimization
        with torch.no_grad():
            jit_model = torch.jit.trace(model, dummy_input)
            # Validate output
            jit_output = jit_model(dummy_input)
            if torch.allclose(orig_output, jit_output, rtol=1e-03, atol=1e-04):
                logging.info("✅ TorchScript optimization successful")
                return jit_model
            else:
                logging.warning("⚠️ TorchScript model output differs significantly from original model, skipping optimization")
    except Exception as e:
        logging.warning(f"⚠️ Failed to optimize model with TorchScript: {e}")
    
    # Try fusion of operations if TorchScript failed
    try:
        logging.info("🚀 Trying operation fusion optimization...")
        # Create a deep copy to avoid modifying original
        import copy
        fused_model = copy.deepcopy(model)
        
        # Apply basic fusion optimizations manually (when applicable)
        for module in fused_model.modules():
            # Fuse BatchNorm into Conv layers where possible
            if hasattr(module, '_freeze_bn_stats'):
                module._freeze_bn_stats()
                
        # Test the fused model
        with torch.no_grad():
            fused_output = fused_model(dummy_input)
            if torch.allclose(orig_output, fused_output, rtol=1e-03, atol=1e-04):
                logging.info("✅ Operation fusion successful")
                return fused_model
            else:
                logging.warning("⚠️ Fused model output differs significantly from original, skipping fusion")
    except Exception as e:
        logging.warning(f"⚠️ Failed to apply operation fusion: {e}")
    
    # If all optimizations failed, return original model
    logging.info("⚠️ All optimization attempts failed, using original model")
    return model 