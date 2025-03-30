import logging
import sys
from pathlib import Path

def setup_logging(dataset, model_name, timestamp, log_level=logging.INFO):
    """
    Set up logging to both console and file
    
    Args:
        dataset: Name of the dataset
        model_name: Name of the model
        timestamp: Timestamp for the experiment
        log_level: Logging level (default: INFO)
    
    Returns:
        Path to the log file
    """
    # Create experiment directory
    experiment_dir = Path("/mnt/d/Ubuntu/phd/Prune_Methods/Restructured_SP/experiments") / dataset
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiment_dir / f"{model_name}_{dataset}_{timestamp}.log"
    
    # Reset the root logger by removing all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure root logger with both file and console handlers
    root_logger.setLevel(log_level)
    
    # Add file handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    logging.info(f"📝 Logging setup complete. Log file: {log_path}")
    return log_path

def log_system_info():
    """Log system information about CUDA availability"""
    import torch
    import os
    import subprocess
    
    # Check if we're running in WSL
    in_wsl = False
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                in_wsl = True
                logging.info("Running in Windows Subsystem for Linux (WSL)")
    except:
        pass

    # Check CUDA environment variables
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        logging.warning("CUDA_VISIBLE_DEVICES not set in environment")

    # Check NVIDIA driver status
    nvidia_driver_status = "Unknown"
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            nvidia_driver_status = "Working"
            logging.info("NVIDIA driver is installed and working")
        else:
            if "access blocked by the operating system" in result.stderr:
                nvidia_driver_status = "Blocked"
                logging.warning("NVIDIA GPU access is blocked by the operating system")
                if in_wsl:
                    logging.warning("In WSL, you need to enable CUDA in WSL2. You may need to:")
                    logging.warning("1. Update your WSL kernel: wsl --update")
                    logging.warning("2. Install NVIDIA drivers for WSL: https://docs.nvidia.com/cuda/wsl-user-guide/index.html")
                    logging.warning("3. Set environment variable in .bashrc: export CUDA_VISIBLE_DEVICES=0")
            else:
                nvidia_driver_status = "Not working"
                logging.warning(f"NVIDIA driver found but not working: {result.stderr}")
    except Exception as e:
        nvidia_driver_status = "Not found"
        logging.warning(f"NVIDIA driver not found: {e}")

    # Configure CUDA settings
    try:
        # Attempt to use CUDA if available
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            cuda_device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            device = torch.device("cuda")
            cuda_available = True
            
            logging.info(f"✅ CUDA is available. Found {cuda_device_count} device(s)")
            logging.info(f"✅ Using GPU: {torch.cuda.get_device_name(current_device)}")
            logging.info(f"✅ GPU Memory: {torch.cuda.get_device_properties(current_device).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device("cpu")
            cuda_available = False
            
            # Log detailed error information
            logging.warning("⚠️ CUDA is not available to PyTorch. Using CPU for computations.")
            logging.warning("⚠️ CPU computation will be SIGNIFICANTLY slower for neural network training.")
            
            # Extra debug information based on driver status
            if nvidia_driver_status == "Blocked":
                if in_wsl:
                    logging.warning("🔍 WSL2 CUDA Issue: GPU is present but access is blocked by WSL.")
                    logging.warning("   Follow the instructions above to enable CUDA in WSL2.")
                else:
                    logging.warning("🔍 GPU access is blocked by the operating system.")
                    logging.warning("   Check if GPU is assigned to another process or virtualization layer.")
            elif nvidia_driver_status == "Working":
                logging.warning("🔍 NVIDIA driver is working but PyTorch can't access CUDA.")
                logging.warning("   Check if PyTorch is installed with CUDA support: pip install torch --upgrade")
                
            # Log PyTorch and CUDA version information
            logging.info(f"PyTorch version: {torch.__version__}")
            # Check CUDA availability in the simplest way
            cuda_compiled = torch.cuda.is_available()
            logging.info(f"PyTorch CUDA available: {cuda_compiled}")
            
            # Try to get CUDA version if available
            try:
                if hasattr(torch._C, '_cuda_getCompiledVersion'):
                    cuda_ver = torch._C._cuda_getCompiledVersion()
                    logging.info(f"PyTorch CUDA compiled version: {cuda_ver}")
            except Exception as e:
                logging.warning(f"Unable to determine CUDA version: {e}")
            
    except Exception as e:
        logging.error(f"❌ Error setting up CUDA: {e}")
        device = torch.device("cpu")
        cuda_available = False
        logging.warning("⚠️ Using CPU due to CUDA setup error.")
        
    return device, cuda_available 