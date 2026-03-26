"""
Image preprocessing and face detection utilities for KYC system
"""

import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detection and alignment using MTCNN"""
    
    def __init__(self, min_face_size=20):
        """
        Initialize face detector
        
        Args:
            min_face_size: Minimum face size to detect
        """
        try:
            from insightface.app import FaceAnalysis
            self.detector = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.detector.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("FaceAnalysis detector initialized")
        except Exception as e:
            logger.warning(f"Could not load insightface: {e}. Falling back to OpenCV Haar Cascade")
            self.detector = None
            # Try different cascade files for better detection
            cascade_files = [
                'haarcascade_frontalface_alt2.xml',
                'haarcascade_frontalface_alt.xml',
                'haarcascade_frontalface_default.xml'
            ]
            self.face_cascade = None
            for cascade_file in cascade_files:
                try:
                    self.face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + cascade_file
                    )
                    if self.face_cascade.empty():
                        continue
                    logger.info(f"Loaded Haar cascade: {cascade_file}")
                    break
                except:
                    continue
            if self.face_cascade is None or self.face_cascade.empty():
                logger.error("Could not load any Haar cascade classifier")
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face bounding box in image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Bounding box (x, y, w, h) or None if no face detected
        """
        if self.detector is not None:
            # Use insightface
            try:
                faces = self.detector.get(image)
                if len(faces) > 0:
                    # Return largest face
                    largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                    bbox = largest_face.bbox.astype(int)
                    return tuple(bbox[:4])
            except Exception as e:
                logger.warning(f"Insightface detection failed: {e}")
                pass
        
        # Fallback to Haar Cascade with very lenient parameters for synthetic faces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Very lenient parameters for synthetic face detection
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.02,  # Very small steps
            minNeighbors=1,    # Very few neighbors required
            minSize=(15, 15),  # Very small minimum size
            maxSize=(400, 400) # Allow larger faces
        )
        
        logger.info(f"Haar Cascade detected {len(faces)} faces")
        if len(faces) > 0:
            # Return largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            logger.info(f"Largest face: {largest}")
            return tuple(largest)
        
        return None
    
    def align_face(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
        """
        Detect and align face in image
        
        Args:
            image: Input image
            target_size: Target output size (height, width)
            
        Returns:
            Aligned face image or None
        """
        bbox = self.detect_face(image)
        if bbox is None:
            logger.warning("No face detected in image")
            return None
        
        x, y, w, h = bbox
        x, y = max(0, x), max(0, y)
        
        # Add padding
        padding = int(0.1 * min(w, h))
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = w + 2*padding, h + 2*padding
        
        # Crop face
        face = image[y:y+h, x:x+w]
        
        if face.size == 0:
            logger.warning("Face crop resulted in empty image")
            return None
        
        # Resize to target size
        face_aligned = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
        
        return face_aligned


class ImagePreprocessor:
    """Image preprocessing and normalization"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image array in BGR format or None
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return None
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to read image: {image_path}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def normalize_image(image: np.ndarray, method: str = 'imagenet') -> np.ndarray:
        """
        Normalize image pixel values
        
        Args:
            image: Input image
            method: Normalization method ('imagenet', 'standard', 'minmax')
            
        Returns:
            Normalized image
        """
        image = image.astype(np.float32)
        
        if method == 'imagenet':
            # ImageNet normalization
            mean = np.array([103.939, 116.779, 123.68])
            image[..., 0] -= mean[0]
            image[..., 1] -= mean[1]
            image[..., 2] -= mean[2]
        elif method == 'standard':
            # Standard normalization [0, 1]
            image = image / 255.0
        elif method == 'minmax':
            # Min-max normalization to [-1, 1]
            image = 2.0 * (image - image.min()) / (image.max() - image.min() + 1e-7) - 1.0
        
        return image
    
    @staticmethod
    def preprocess_face_pair(
        image1_path: str,
        image2_path: str,
        target_size: Tuple[int, int] = (224, 224),
        normalize_method: str = 'imagenet'
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess pair of images (selfie and ID photo)
        
        Args:
            image1_path: Path to first image
            image2_path: Path to second image
            target_size: Target output size
            normalize_method: Normalization method
            
        Returns:
            Tuple of preprocessed images or None
        """
        detector = FaceDetector()
        
        # Load and process first image
        img1 = ImagePreprocessor.load_image(image1_path)
        if img1 is None:
            return None
        
        face1 = detector.align_face(img1, target_size)
        if face1 is None:
            logger.error(f"Could not process face in {image1_path}")
            return None
        
        face1 = ImagePreprocessor.normalize_image(face1, normalize_method)
        
        # Load and process second image
        img2 = ImagePreprocessor.load_image(image2_path)
        if img2 is None:
            return None
        
        face2 = detector.align_face(img2, target_size)
        if face2 is None:
            logger.error(f"Could not process face in {image2_path}")
            return None
        
        face2 = ImagePreprocessor.normalize_image(face2, normalize_method)
        
        return face1, face2


def create_sample_data():
    """Create sample data directory structure"""
    os.makedirs('data/train/same', exist_ok=True)
    os.makedirs('data/train/different', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    logger.info("Sample data directories created")


if __name__ == "__main__":
    logger.info("KYC Utils module loaded successfully")
    create_sample_data()
