"""
Demo script for KYC Identity Verification System
Shows how to use the system with sample code
"""

def example_1_basic_verification():
    """Example 1: Basic identity verification"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Identity Verification")
    print("="*70)
    
    from main import KYCVerificationSystem
    
    # Initialize the KYC system
    kyc = KYCVerificationSystem()
    
    # Verify images
    result = kyc.verify_identity('data/test/selfie.jpg', 'data/test/id_card.jpg')
    
    # Display result
    kyc.print_result(result)
    
    return result


def example_2_batch_verification():
    """Example 2: Batch verification of multiple image pairs"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Verification of Multiple Pairs")
    print("="*70)
    
    from main import KYCVerificationSystem
    import os
    
    kyc = KYCVerificationSystem()
    
    # List of (selfie, id_photo) pairs to verify
    pairs = [
        ('data/test/person1_selfie.jpg', 'data/test/person1_id.jpg'),
        ('data/test/person2_selfie.jpg', 'data/test/person2_id.jpg'),
        ('data/test/person3_selfie.jpg', 'data/test/person3_id.jpg'),
    ]
    
    results = []
    for selfie, id_photo in pairs:
        if os.path.exists(selfie) and os.path.exists(id_photo):
            result = kyc.verify_identity(selfie, id_photo)
            results.append(result)
            print(f"\n{os.path.basename(selfie)}: {result['status']}")
    
    return results


def example_3_threshold_tuning():
    """Example 3: Adjust verification threshold for different security levels"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Threshold Tuning for Different Security Levels")
    print("="*70)
    
    from main import KYCVerificationSystem
    
    kyc = KYCVerificationSystem()
    selfie = 'data/test/selfie.jpg'
    id_photo = 'data/test/id_card.jpg'
    
    # Test with different thresholds
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("\nTesting same pair with different thresholds:")
    print("-" * 70)
    
    for threshold in thresholds:
        kyc.set_threshold(threshold)
        result = kyc.verify_identity(selfie, id_photo)
        
        if result['distance'] is not None:
            security_level = "STRICT" if threshold <= 0.5 else "BALANCED" if threshold <= 0.6 else "LENIENT"
            print(f"Threshold {threshold:.1f} ({security_level:8}): {result['status']:8} (dist={result['distance']:.4f}, conf={result['confidence']:.1%})")
    
    # Reset to default
    kyc.set_threshold(0.6)


def example_4_configuration():
    """Example 4: Access and modify configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Configuration Management")
    print("="*70)
    
    from config import MODEL_CONFIG, VERIFICATION_CONFIG, TRAINING_CONFIG, DATA_CONFIG
    
    print("\nModel Configuration:")
    print(f"  - Input shape: {MODEL_CONFIG['input_shape']}")
    print(f"  - Embedding dimension: {MODEL_CONFIG['embedding_dim']}")
    print(f"  - Base model: {MODEL_CONFIG['base_model']}")
    
    print("\nVerification Configuration:")
    print(f"  - Distance threshold: {VERIFICATION_CONFIG['distance_threshold']}")
    print(f"  - Distance metric: {VERIFICATION_CONFIG['metric']}")
    
    print("\nTraining Configuration (for future training):")
    print(f"  - Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"  - Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"  - Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"  - Loss function: {TRAINING_CONFIG['loss_function']}")
    
    print("\nData Paths:")
    print(f"  - Training data: {DATA_CONFIG['train_path']}")
    print(f"  - Test data: {DATA_CONFIG['test_path']}")
    print(f"  - Model save path: {DATA_CONFIG['model_save_path']}")


def example_5_image_preprocessing():
    """Example 5: Image preprocessing and face detection"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Image Preprocessing & Face Detection")
    print("="*70)
    
    from utils import ImagePreprocessor, FaceDetector
    import cv2
    
    # Load an image
    image_path = 'data/test/selfie.jpg'
    
    # Initialize detector
    detector = FaceDetector()
    image = ImagePreprocessor.load_image(image_path)
    
    if image is not None:
        # Detect face
        bbox = detector.detect_face(image)
        if bbox:
            print(f"\n✓ Face detected at bounding box: {bbox}")
            print(f"  - X: {bbox[0]}, Y: {bbox[1]}")
            print(f"  - Width: {bbox[2]}, Height: {bbox[3]}")
            
            # Align face
            aligned_face = detector.align_face(image)
            if aligned_face is not None:
                print(f"✓ Face aligned to size: {aligned_face.shape}")
                
                # Normalize
                normalized = ImagePreprocessor.normalize_image(aligned_face, method='imagenet')
                print(f"✓ Face normalized (ImageNet method)")
                print(f"  - Pixel range: [{normalized.min():.2f}, {normalized.max():.2f}]")
        else:
            print("✗ No face detected in image")
    else:
        print(f"✗ Could not load image: {image_path}")


def example_6_model_architecture():
    """Example 6: Inspect model architecture"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Model Architecture Inspection")
    print("="*70)
    
    from model import SiameseNetwork
    
    # Build model
    siamese = SiameseNetwork(input_shape=(224, 224, 3), embedding_dim=128)
    siamese.build_siamese_model()
    
    print("\n📊 Siamese Network Architecture:")
    print("-" * 70)
    
    # Get models
    embedding_model = siamese.get_embedding_model()
    siamese_model = siamese.get_siamese_model()
    
    print(f"\nEmbedding Model (Face Feature Extraction):")
    print(f"  - Input: (224, 224, 3) RGB images")
    print(f"  - Output: 128-dimensional L2-normalized embeddings")
    print(f"  - Total parameters: {embedding_model.count_params():,}")
    
    print(f"\nSiamese Model (Verification):")
    print(f"  - Input A: (224, 224, 3) Selfie")
    print(f"  - Input B: (224, 224, 3) ID Photo")
    print(f"  - Output: Similarity score (0.0-1.0)")
    print(f"  - Total parameters: {siamese_model.count_params():,}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*70)
    print("KYC Identity Verification System - Demo Examples")
    print("="*70)
    print("\nThese examples show how to use the KYC system in your code.")
    print("Make sure to have test images in data/test/ folder first.\n")
    
    examples = {
        '1': ('Basic Verification', example_1_basic_verification),
        '2': ('Batch Verification', example_2_batch_verification),
        '3': ('Threshold Tuning', example_3_threshold_tuning),
        '4': ('Configuration', example_4_configuration),
        '5': ('Image Preprocessing', example_5_image_preprocessing),
        '6': ('Model Architecture', example_6_model_architecture),
    }
    
    print("Available Examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}: {name}")
    
    print("\nUsage:")
    print("  python demo.py <example_number>")
    print("  python demo.py 1          # Run Example 1")
    print("  python demo.py all        # Run all examples\n")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1].lower()
        
        if choice == 'all':
            for key, (name, func) in examples.items():
                try:
                    func()
                except Exception as e:
                    print(f"\n✗ Error in {name}: {e}")
        elif choice in examples:
            name, func = examples[choice]
            try:
                func()
            except Exception as e:
                print(f"\n✗ Error in {name}: {e}")
        else:
            print(f"✗ Invalid example number: {choice}")
            print(f"   Valid options: {', '.join(examples.keys())} or 'all'")
    else:
        print("Run 'python demo.py <number>' to execute an example.\n")
