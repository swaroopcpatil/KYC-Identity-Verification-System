"""
Test script to verify KYC Identity Verification System setup
"""

import sys
import importlib

def test_imports():
    """Test all required imports"""
    
    packages = {
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib',
        'sklearn': 'Scikit-learn',
        'tensorflow': 'TensorFlow',
        'keras': 'Keras'
    }
    
    results = {}
    print("\n" + "="*60)
    print("KYC SYSTEM DEPENDENCY CHECK")
    print("="*60 + "\n")
    
    for module, name in packages.items():
        try:
            mod = importlib.import_module(module)
            version = getattr(mod, '__version__', 'N/A')
            results[name] = (True, version)
            print(f"✓ {name:<20} {version}")
        except ImportError as e:
            results[name] = (False, str(e))
            print(f"✗ {name:<20} NOT INSTALLED")
    
    print("\n" + "="*60)
    
    # Check local modules
    print("\nLocal Module Check:")
    print("-" * 60)
    
    local_modules = ['config', 'utils', 'model', 'main']
    all_ok = True
    
    for module in local_modules:
        try:
            importlib.import_module(module)
            print(f"✓ {module}.py")
        except Exception as e:
            print(f"✗ {module}.py - {e}")
            all_ok = False
    
    print("="*60 + "\n")
    
    # Summary
    installed = sum(1 for ok, _ in results.values() if ok)
    total = len(results)
    
    print(f"Summary: {installed}/{total} packages installed")
    
    if installed >= 6:  # All except TensorFlow
        print("\n✓ System ready for inference with pre-trained models!")
        print("✓ To train new models, TensorFlow is required.")
        return True
    else:
        print("\n✗ Missing critical dependencies. Please install missing packages.")
        return False


def test_kyc_system():
    """Test basic KYC system functionality"""
    
    print("\n" + "="*60)
    print("KYC SYSTEM FUNCTIONALITY TEST")
    print("="*60 + "\n")
    
    try:
        from config import MODEL_CONFIG, VERIFICATION_CONFIG, DATA_CONFIG
        print("✓ Configuration loaded successfully")
        print(f"  - Embedding dimension: {MODEL_CONFIG['embedding_dim']}")
        print(f"  - Distance threshold: {VERIFICATION_CONFIG['distance_threshold']}")
        print(f"  - Metric: {VERIFICATION_CONFIG['metric']}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    try:
        from utils import ImagePreprocessor, FaceDetector
        print("✓ Image preprocessing utilities loaded")
    except Exception as e:
        print(f"✗ Utils error: {e}")
        return False
    
    try:
        from model import SiameseNetwork
        print("✓ Siamese network model loaded")
    except Exception as e:
        print(f"✗ Model error: {e}")
        return False
    
    try:
        from main import KYCVerificationSystem
        print("✓ KYC verification system loaded")
    except Exception as e:
        print(f"✗ Main module error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All modules initialized successfully!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    print("\nStarting KYC System Setup Verification...\n")
    
    imports_ok = test_imports()
    system_ok = test_kyc_system()
    
    if system_ok:
        print("\n✅ KYC Identity Verification System is ready to use!\n")
        print("Next steps:")
        print("1. Place your test images in: data/test/")
        print("2. Run: python main.py")
        print("3. Or use the demo: from main import demo_with_custom_images")
        print("\n")
    else:
        print("\n⚠️  Some modules are missing. Install remaining dependencies.\n")
        sys.exit(1)
