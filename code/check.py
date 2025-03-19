import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
import sys
import subprocess

def check_cuda_installation():
    """
    Comprehensive check of PyTorch CUDA installation and configuration
    Returns detailed diagnostic information
    """
    results = {
        "System Information": {},
        "PyTorch Installation": {},
        "CUDA Configuration": {},
        "GPU Information": {},
    }
    
    # System Information
    results["System Information"]["Python Version"] = sys.version
    results["System Information"]["Operating System"] = sys.platform
    
    # PyTorch Installation
    results["PyTorch Installation"]["PyTorch Version"] = torch.__version__
    results["PyTorch Installation"]["CUDA Available"] = torch.cuda.is_available()
    results["PyTorch Installation"]["cuDNN Version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "Not available"
    
    # CUDA Configuration
    if torch.cuda.is_available():
        results["CUDA Configuration"]["CUDA Version"] = torch.version.cuda
        results["CUDA Configuration"]["Device Count"] = torch.cuda.device_count()
        results["CUDA Configuration"]["Current Device"] = torch.cuda.current_device()
        results["CUDA Configuration"]["Device Name"] = torch.cuda.get_device_name(0)
    
    # Environment Variables
    cuda_path = os.environ.get("CUDA_PATH", "Not set")
    cuda_home = os.environ.get("CUDA_HOME", "Not set")
    results["CUDA Configuration"]["CUDA_PATH"] = cuda_path
    results["CUDA Configuration"]["CUDA_HOME"] = cuda_home
    
    # GPU Information using nvidia-smi if available
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        results["GPU Information"]["nvidia-smi"] = nvidia_smi
    except (subprocess.SubprocessError, FileNotFoundError):
        results["GPU Information"]["nvidia-smi"] = "nvidia-smi not available or failed to run"
    
    return results

def print_diagnostic_results(results):
    """
    Print the diagnostic results in a readable format
    """
    print("\n=== PyTorch CUDA Installation Diagnostic Report ===\n")
    
    for section, data in results.items():
        print(f"\n{section}:")
        print("-" * (len(section) + 1))
        for key, value in data.items():
            if isinstance(value, str) and "\n" in value:
                print(f"\n{key}:\n{value}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    results = check_cuda_installation()
    print_diagnostic_results(results)
    
    # Additional helpful information
    print("\n=== Quick CUDA Test ===")
    if torch.cuda.is_available():
        try:
            # Try to create a tensor on GPU
            test_tensor = torch.tensor([1.0, 2.0], device="cuda")
            print("✓ Successfully created tensor on GPU")
            print(f"  Tensor device: {test_tensor.device}")
        except RuntimeError as e:
            print("✗ Failed to create tensor on GPU")
            print(f"  Error: {str(e)}")
    else:
        print("✗ CUDA is not available")