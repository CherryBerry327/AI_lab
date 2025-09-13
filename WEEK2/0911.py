try:
    import torch
    print("✓ PyTorch is successfully imported!")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA is available! Device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Create a simple tensor to test basic functionality
    # tensor를 사용할 때 device나 data type을 지정해줄 수 있고, device만 지정해주었다. 
    test_tensor = torch.tensor([1, 2, 3, 4, 5],device='cuda') 

    print(f"✓ Test tensor created: {test_tensor}")
    print(f"Tensor shape: {test_tensor.shape}")
    print(f"Tensor device: {test_tensor.device}")
    
    # Basic tensor operation
    squared_tensor = test_tensor ** 2
    print(f"✓ Basic operation test (squaring): {squared_tensor}")
    
    print("\n🎉 PyTorch is working correctly!")
    
except ImportError as e:
    print("❌ PyTorch is not installed or not available")
    print(f"Error details: {e}")
    print("Please install PyTorch using: pip install torch")
    
except Exception as e:
    print(f"❌ An error occurred while testing PyTorch: {e}")