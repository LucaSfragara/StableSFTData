import torch
import math

def calculate_model_size(n_params: int, dtype=torch.float16):
    """Calculate memory requirements for n parameters"""
    bytes_per_param = torch.tensor([], dtype=dtype).element_size()
    
    # Memory for parameters
    param_memory = n_params * bytes_per_param
    
    # Optimizer states (if training)
    optimizer_memory = param_memory * 2  # Adam requires 2 states per param
    
    # Convert to GB
    total_gb = (param_memory + optimizer_memory) / (1024**3)
    return total_gb

def test_h200_capacity():
    H200_MEMORY = 141  # GB
    
    # Test different model sizes
    sizes = [
        1e9,   # 1B parameters
        10e9,  # 10B parameters
        70e9,  # 70B parameters
        100e9  # 100B parameters
    ]
    
    print("\nH200 GPU Memory Capacity Analysis")
    print("-" * 50)
    print(f"Available Memory: {H200_MEMORY}GB")
    print("\nModel Size Analysis (FP16):")
    
    for n_params in sizes:
        memory_needed = calculate_model_size(int(n_params))
        fits = memory_needed < H200_MEMORY
        print(f"\n{n_params/1e9:.1f}B parameters:")
        print(f"Memory required: {memory_needed:.1f}GB")
        print(f"Fits in memory: {fits}")

if __name__ == "__main__":
    test_h200_capacity()