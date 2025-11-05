import sys
try:
    import torch
    print(' PyTorch installed')
    print(f'  Version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'\nGPU available: {torch.cuda.get_device_name(0)}')
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  GPU count: {torch.cuda.device_count()}')
        print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1000000000.0:.2f} GB')
        x = torch.randn(1000, 1000).cuda()
        y = x @ x.T
        print(f'\nGPU test successful')
        print(f'  Can move tensors to GPU and perform operations')
    else:
        print('\n GPU NOT available')
        print('  PyTorch will use CPU (slower)')
        print('\nTo enable GPU:')
        print('  1. Ensure NVIDIA GPU drivers are installed')
        print('  2. Install CUDA toolkit')
        print('  3. Install PyTorch with CUDA:')
        print('     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
except ImportError:
    print(' PyTorch NOT installed')
    print('\nInstall with:')
    print('  For GPU (CUDA 11.8):')
    print('    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118')
    print('  For GPU (CUDA 12.1):')
    print('    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
    print('  For CPU only:')
    print('    pip install torch torchvision torchaudio')
    sys.exit(1)
