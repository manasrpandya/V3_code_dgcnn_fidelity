"""
Validation script to check if everything is set up correctly
Run this before starting training to ensure all components work
"""

import torch
import sys
import os
from pathlib import Path


def print_section(title):
    """Print formatted section header"""
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}\n")


def check_imports():
    """Check if all required packages are installed"""
    print_section("Checking Dependencies")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"✓ {name:<20} installed")
        except ImportError:
            print(f"✗ {name:<20} NOT installed")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability"""
    print_section("Checking GPU")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        return True
    else:
        print(f"⚠ CUDA not available (will use CPU)")
        print(f"  Training will be slower on CPU")
        return False


def check_data():
    """Check if preprocessed data exists"""
    print_section("Checking Data")
    
    data_dir = Path('./processed_chunks')
    
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        print(f"  Please ensure processed_chunks/ exists with preprocessed data")
        return False
    
    print(f"✓ Data directory found: {data_dir}")
    
    # Check sessions
    sessions = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('Session')]
    
    if not sessions:
        print(f"✗ No session directories found")
        return False
    
    print(f"\n✓ Found {len(sessions)} session(s):")
    
    total_files = 0
    for session in sorted(sessions):
        npz_files = list(session.glob('*.npz'))
        total_files += len(npz_files)
        print(f"  {session.name}: {len(npz_files)} files")
    
    print(f"\nTotal samples: {total_files}")
    
    if total_files == 0:
        print(f"✗ No .npz files found in sessions")
        return False
    
    return True


def check_model_files():
    """Check if model files exist"""
    print_section("Checking Model Files")
    
    required_files = [
        'models_fidelity_dgcnn.py',
        'iemocap_dataset.py',
        'train_full.py',
    ]
    
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file:<30} found")
        else:
            print(f"✗ {file:<30} NOT found")
            all_ok = False
    
    return all_ok


def test_data_loading():
    """Test loading a sample from the dataset"""
    print_section("Testing Data Loading")
    
    try:
        from iemocap_dataset import IEMOCAPPreprocessedDataset
        
        # Try to create dataset
        dataset = IEMOCAPPreprocessedDataset(
            data_dir='./processed_chunks',
            sessions=['Session1']
        )
        
        if len(dataset) == 0:
            print(f"✗ Dataset is empty")
            return False
        
        print(f"✓ Dataset created successfully")
        print(f"  Total samples: {len(dataset)}")
        
        # Try to load a sample
        text, audio, video, label = dataset[0]
        
        print(f"\n✓ Sample loaded successfully:")
        print(f"  Text shape:  {text.shape} (expected: [768, 20])")
        print(f"  Audio shape: {audio.shape} (expected: [40, 20])")
        print(f"  Video shape: {video.shape} (expected: [2048, 20])")
        print(f"  Label: {label}")
        
        # Verify shapes
        if text.shape != (768, 20):
            print(f"  ⚠ Warning: Text shape mismatch")
        if audio.shape != (40, 20):
            print(f"  ⚠ Warning: Audio shape mismatch")
        if video.shape != (2048, 20):
            print(f"  ⚠ Warning: Video shape mismatch")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_initialization():
    """Test model initialization"""
    print_section("Testing Model Initialization")
    
    try:
        from models_fidelity_dgcnn import FidelityAwareMultimodalDGCNN
        
        # Create dummy hyperparameters
        class HyperParams:
            def __init__(self):
                self.orig_d_l = 768    # BERT
                self.orig_d_a = 40     # MFCC
                self.orig_d_v = 2048   # ResNet-50
                self.output_dim = 4    # 4 emotions
        
        hyp_params = HyperParams()
        model = FidelityAwareMultimodalDGCNN(hyp_params)
        
        print(f"✓ Model created successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create dummy inputs
        batch_size = 2
        text = torch.randn(batch_size, 768, 20).to(device)
        audio = torch.randn(batch_size, 40, 20).to(device)
        video = torch.randn(batch_size, 2048, 20).to(device)
        
        with torch.no_grad():
            output = model(text, audio, video)
        
        print(f"\n✓ Forward pass successful")
        print(f"  Output shape: {output.shape} (expected: [2, 4])")
        
        if output.shape != (batch_size, 4):
            print(f"  ⚠ Warning: Output shape mismatch")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_dir():
    """Check if checkpoint directory exists"""
    print_section("Checking Checkpoint Directory")
    
    checkpoint_dir = Path('./checkpoints')
    
    if not checkpoint_dir.exists():
        print(f"⚠ Checkpoint directory not found")
        print(f"  Creating: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created checkpoint directory")
    else:
        print(f"✓ Checkpoint directory exists")
    
    # Check if writable
    try:
        test_file = checkpoint_dir / '.test'
        test_file.touch()
        test_file.unlink()
        print(f"✓ Checkpoint directory is writable")
        return True
    except Exception as e:
        print(f"✗ Checkpoint directory is not writable: {e}")
        return False


def main():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("Fidelity-Aware DGCNN Setup Validation".center(70))
    print("="*70)
    
    results = {
        'Dependencies': check_imports(),
        'GPU': check_gpu(),
        'Data': check_data(),
        'Model Files': check_model_files(),
        'Data Loading': test_data_loading(),
        'Model Init': test_model_initialization(),
        'Checkpoint Dir': test_checkpoint_dir(),
    }
    
    # Summary
    print_section("Validation Summary")
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:<20} {status}")
    
    # GPU is optional, so check critical components only
    critical_checks = {k: v for k, v in results.items() if k != 'GPU'}
    all_passed = all(critical_checks.values())
    gpu_available = results['GPU']
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ All critical checks passed! You're ready to train.".center(70))
        print(f"{'='*70}\n")
        print("To start training, run:")
        print("  python train_full.py --data_dir ./processed_chunks --sessions Session1")
        if not gpu_available:
            print("\n⚠ Note: No GPU detected. Training will use CPU (slower).")
            print("  To force CPU explicitly: python train_full.py --device cpu")
        else:
            print("\n✓ GPU is available! Training will be faster.")
    else:
        print("✗ Critical checks failed. Please fix the issues above.".center(70))
        print(f"{'='*70}\n")
        sys.exit(1)
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
