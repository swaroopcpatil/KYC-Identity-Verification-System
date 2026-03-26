"""
Main script for KYC Identity Verification System
Performs 1:1 face verification between selfie and ID photo
"""

import os
import sys
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import custom modules
from config import VERIFICATION_CONFIG, MODEL_CONFIG, DATA_CONFIG
from utils import ImagePreprocessor, FaceDetector
from model import SiameseNetwork

import tensorflow as tf


class KYCVerificationSystem:
    """KYC Identity Verification System"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize KYC system
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model_path = model_path
        self.siamese_network = None
        self.embedding_model = None
        self.distance_threshold = VERIFICATION_CONFIG['distance_threshold']
        self.metric = VERIFICATION_CONFIG['metric']
        
        logger.info("KYC Verification System initialized")
    
    def load_model(self):
        """Load or build the model"""
        if self.model_path and os.path.exists(self.model_path):
            logger.info(f"Loading pre-trained model from {self.model_path}")
            try:
                self.embedding_model = tf.keras.models.load_model(self.model_path)
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Building new model...")
                self._build_model()
        else:
            logger.info("Building new Siamese network model")
            self._build_model()
    
    def _build_model(self):
        """Build new model"""
        siamese = SiameseNetwork(
            input_shape=tuple(MODEL_CONFIG['input_shape']),
            embedding_dim=MODEL_CONFIG['embedding_dim']
        )
        siamese.build_siamese_model()
        self.embedding_model = siamese.get_embedding_model()
        logger.info("New model built successfully")
    
    def extract_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image
        
        Args:
            image: Preprocessed face image
            
        Returns:
            Face embedding vector or None
        """
        if self.embedding_model is None:
            self.load_model()
        
        try:
            # Add batch dimension
            image_batch = np.expand_dims(image, axis=0)
            embedding = self.embedding_model.predict(image_batch, verbose=0)
            return embedding[0]
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None
    
    def calculate_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate distance between two embeddings
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Distance value
        """
        if self.metric == 'euclidean':
            distance = np.sqrt(np.sum(np.square(embedding1 - embedding2)))
        elif self.metric == 'cosine':
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            distance = 1 - (dot_product / (norm1 * norm2 + 1e-7))
        else:
            distance = np.sqrt(np.sum(np.square(embedding1 - embedding2)))
        
        return float(distance)
    
    def verify_identity(self, selfie_path: str, id_photo_path: str) -> Dict:
        """
        Verify identity by comparing selfie to ID photo
        
        Args:
            selfie_path: Path to selfie image
            id_photo_path: Path to ID photo
            
        Returns:
            Dictionary with verification result
        """
        logger.info(f"Starting verification: {selfie_path} vs {id_photo_path}")
        
        # Preprocess images
        result = ImagePreprocessor.preprocess_face_pair(
            selfie_path,
            id_photo_path,
            target_size=tuple(MODEL_CONFIG['input_shape'][:2]),
            normalize_method='imagenet'
        )
        
        if result is None:
            logger.error("Failed to preprocess images")
            return {
                'status': 'REJECTED',
                'reason': 'Face detection failed',
                'distance': None,
                'confidence': 0.0
            }
        
        face1, face2 = result
        
        # Extract embeddings
        embedding1 = self.extract_embedding(face1)
        embedding2 = self.extract_embedding(face2)
        
        if embedding1 is None or embedding2 is None:
            logger.error("Failed to extract embeddings")
            return {
                'status': 'REJECTED',
                'reason': 'Embedding extraction failed',
                'distance': None,
                'confidence': 0.0
            }
        
        # Calculate distance
        distance = self.calculate_distance(embedding1, embedding2)
        
        # Determine verification result
        is_verified = distance < self.distance_threshold
        confidence = 1.0 - (distance / max(self.distance_threshold * 2, 1.0))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        
        status = 'VERIFIED' if is_verified else 'REJECTED'
        
        result_dict = {
            'status': status,
            'distance': float(distance),
            'threshold': self.distance_threshold,
            'confidence': float(confidence),
            'metric': self.metric,
            'selfie_path': selfie_path,
            'id_photo_path': id_photo_path
        }
        
        logger.info(f"Verification result: {status} (distance: {distance:.4f}, confidence: {confidence:.2%})")
        
        return result_dict
    
    def set_threshold(self, threshold: float):
        """
        Set distance threshold for verification
        
        Args:
            threshold: New threshold value
        """
        self.distance_threshold = threshold
        logger.info(f"Distance threshold set to {threshold}")
    
    def print_result(self, result: Dict):
        """
        Print verification result in formatted way
        
        Args:
            result: Result dictionary from verify_identity
        """
        print("\n" + "="*60)
        print("KYC IDENTITY VERIFICATION RESULT")
        print("="*60)
        print(f"Status:        {result['status']}")
        if result['distance'] is not None:
            print(f"Distance:      {result['distance']:.4f}")
            print(f"Threshold:     {result['threshold']:.4f}")
            print(f"Confidence:    {result['confidence']:.2%}")
            print(f"Metric:        {result['metric']}")
        else:
            print(f"Reason:        {result.get('reason', 'Unknown error')}")
        if 'selfie_path' in result:
            print(f"Selfie:        {result['selfie_path']}")
        if 'id_photo_path' in result:
            print(f"ID Photo:      {result['id_photo_path']}")
        print("="*60 + "\n")


def main():
    """Main execution function"""
    
    logger.info("Starting KYC Identity Verification System")
    
    # Initialize system
    kyc_system = KYCVerificationSystem(model_path=DATA_CONFIG['model_save_path'])
    
    # Example usage - verify test images
    print("\nKYC Identity Verification System")
    print("-" * 40)
    
    # Check if test images exist
    test_selfie = "data/test/selfie.jpg"
    test_id = "data/test/id_card.jpg"
    
    if not os.path.exists(test_selfie) or not os.path.exists(test_id):
        print("\nNo test images found. Creating demo...")
        print(f"Please place test images at:")
        print(f"  - Selfie: {test_selfie}")
        print(f"  - ID Photo: {test_id}")
        print("\nOr use verify_identity() function directly in your code:")
        print("\n  result = kyc_system.verify_identity('path/to/selfie.jpg', 'path/to/id.jpg')")
        print("  kyc_system.print_result(result)")
        return
    
    # Run verification
    result = kyc_system.verify_identity(test_selfie, test_id)
    kyc_system.print_result(result)


def demo_with_custom_images(selfie_path: str, id_photo_path: str, threshold: float = 0.6):
    """
    Run KYC verification with custom images
    
    Args:
        selfie_path: Path to selfie image
        id_photo_path: Path to ID photo
        threshold: Distance threshold for verification
    """
    logger.info("Running KYC demo with custom images")
    
    # Initialize system
    kyc_system = KYCVerificationSystem()
    kyc_system.set_threshold(threshold)
    
    # Verify identity
    result = kyc_system.verify_identity(selfie_path, id_photo_path)
    kyc_system.print_result(result)
    
    return result


if __name__ == "__main__":
    main()
