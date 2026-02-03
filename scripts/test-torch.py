import torch
import time

def memory_stats():
    print(f"\n{torch.cuda.memory_allocated()/1024**2}")
    print(torch.cuda.memory_reserved()/1024**2)

def run_gpu_test():
    print("--- PyTorch CUDA Test ---")
    print("6 run")
    import socket
    print(socket.gethostname())
    # 1. Check if CUDA (GPU support) is available
    if torch.cuda.is_available():
        device_id = 0
        device = torch.device(f"cuda:{device_id}")
        print(f"✅ CUDA is available!")
        print(f"   Device Name: {torch.cuda.get_device_name(device_id)}")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA is not available. Running on CPU.")
        device = torch.device("cpu")

    print(f"\nTarget Device: {device}")

    # 2. Create Random Tensors
    # We create two matrices of size 1000x1000
    size = 10000
    print(f"Creating two {size}x{size} random tensors...")

    # 3. Perform a Matrix Multiplication
    print("Performing Matrix Multiplication...")
    start_time = time.time()

    # Create directly on the device (faster than creating on CPU then moving)
    for x in range(2):
        x = torch.rand(size, size, device=device)
        y = torch.rand(size, size, device=device)


        # This operation happens entirely on the GPU (if available)
        z = torch.matmul(x, y)

        # Synchronize assumes async execution (common in CUDA) to get accurate timing
        if device.type == 'cuda':
            torch.cuda.synchronize()

    end_time = time.time()

    # 4. Verify Location and Result
    print("\n--- Results ---")
    print(f"Operation completed in: {end_time - start_time:.4f} seconds")
    print(f"Result Tensor Device: {z.device}")
    print(f"Result Tensor Size: {z.shape}")
    print("Test Complete.")

    import gc

    # memory_stats()

    # device.cpu()
    del device, x, y, z
    gc.collect()
    torch.cuda.empty_cache()

    # memory_stats()

if __name__ == "__main__":
    run_gpu_test()
