"""
Real-Time KYC Identity Verification System
Uses webcam for live verification against a pre-loaded ID photo
"""

import cv2
import numpy as np
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules
from config import MODEL_CONFIG, VERIFICATION_CONFIG
from utils import FaceDetector, ImagePreprocessor
from model import SiameseNetwork
import tensorflow as tf


class RealTimeKYC:
    """Real-time KYC verification using webcam"""

    def __init__(self, id_photo_path: str):
        """
        Initialize real-time KYC system

        Args:
            id_photo_path: Path to ID photo to verify against
        """
        self.id_photo_path = id_photo_path
        self.distance_threshold = VERIFICATION_CONFIG['distance_threshold']

        # Initialize components
        self.face_detector = FaceDetector()
        self.image_preprocessor = ImagePreprocessor()

        # Build and load model
        logger.info("Building Siamese network model...")
        self.siamese_network = SiameseNetwork(
            input_shape=tuple(MODEL_CONFIG['input_shape']),
            embedding_dim=MODEL_CONFIG['embedding_dim']
        )
        self.siamese_network.build_siamese_model()
        self.embedding_model = self.siamese_network.get_embedding_model()

        # Load and process ID photo
        self._load_id_embedding()

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Make sure it's connected.")

        logger.info("Real-time KYC system initialized successfully")

    def _load_id_embedding(self):
        """Load ID photo and extract its embedding"""
        logger.info(f"Loading ID photo: {self.id_photo_path}")

        # Load image
        id_image = cv2.imread(self.id_photo_path)
        if id_image is None:
            raise ValueError(f"Could not load ID image: {self.id_photo_path}")

        # Detect face
        bbox = self.face_detector.detect_face(id_image)
        if bbox is None:
            raise ValueError("No face detected in ID photo. Please use a clearer image.")

        # Align and preprocess face
        aligned_face = self.face_detector.align_face(id_image, target_size=(224, 224))
        if aligned_face is None:
            raise ValueError("Could not align face in ID photo.")

        # Normalize
        normalized_face = self.image_preprocessor.normalize_image(aligned_face)

        # Extract embedding
        embedding = self._extract_embedding(normalized_face)
        if embedding is None:
            raise ValueError("Could not extract embedding from ID photo.")

        self.id_embedding = embedding
        logger.info("ID photo processed and embedding extracted")

    def _extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """Extract face embedding from preprocessed face image"""
        try:
            # Add batch dimension
            face_batch = np.expand_dims(face_image, axis=0)
            embedding = self.embedding_model.predict(face_batch, verbose=0)
            return embedding[0]
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None

    def verify_frame(self, frame: np.ndarray) -> dict:
        """
        Verify a frame from webcam against the ID photo

        Args:
            frame: Webcam frame (BGR format)

        Returns:
            Dictionary with verification results
        """
        # Detect face in frame
        bbox = self.face_detector.detect_face(frame)
        if bbox is None:
            return {'status': 'NO_FACE', 'distance': None, 'confidence': 0.0, 'bbox': None}

        # Align and preprocess face
        aligned_face = self.face_detector.align_face(frame, target_size=(224, 224))
        if aligned_face is None:
            return {'status': 'FACE_ERROR', 'distance': None, 'confidence': 0.0, 'bbox': bbox}

        # Normalize
        normalized_face = self.image_preprocessor.normalize_image(aligned_face)

        # Extract embedding
        embedding = self._extract_embedding(normalized_face)
        if embedding is None:
            return {'status': 'EMBEDDING_ERROR', 'distance': None, 'confidence': 0.0, 'bbox': bbox}

        # Calculate distance
        distance = np.linalg.norm(self.id_embedding - embedding)

        # Determine result
        is_verified = distance < self.distance_threshold
        confidence = 1.0 - (distance / max(self.distance_threshold * 2, 1.0))
        confidence = max(0.0, min(1.0, confidence))

        status = 'VERIFIED' if is_verified else 'REJECTED'

        return {
            'status': status,
            'distance': float(distance),
            'confidence': float(confidence),
            'bbox': bbox
        }

    def run(self):
        """Run real-time verification"""
        print("\n" + "="*60)
        print("REAL-TIME KYC IDENTITY VERIFICATION")
        print("="*60)
        print(f"ID Photo: {self.id_photo_path}")
        print(f"Threshold: {self.distance_threshold}")
        print("Press 'q' to quit, 'c' to capture current frame")
        print("="*60)

        frame_count = 0
        last_result = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam")
                break

            frame_count += 1

            # Verify current frame
            result = self.verify_frame(frame)
            last_result = result

            # Draw results on frame
            self._draw_results(frame, result)

            # Display frame
            cv2.imshow('Real-Time KYC Verification', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture and save current frame
                capture_path = f"capture_{frame_count}.jpg"
                cv2.imwrite(capture_path, frame)
                print(f"Frame captured: {capture_path}")

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        # Final result
        if last_result:
            print("\nFinal Result:")
            print(f"Status: {last_result['status']}")
            if last_result['distance'] is not None:
                print(f"Distance: {last_result['distance']:.4f}")
                print(f"Confidence: {last_result['confidence']:.2%}")

    def _draw_results(self, frame: np.ndarray, result: dict):
        """Draw verification results on the frame"""
        height, width = frame.shape[:2]

        # Draw bounding box if face detected
        if result['bbox'] is not None:
            x, y, w, h = result['bbox']
            if result['status'] == 'VERIFIED':
                color = (0, 255, 0)  # Green
                text = f"VERIFIED ({result['confidence']:.1f}%)"
            elif result['status'] == 'REJECTED':
                color = (0, 0, 255)  # Red
                text = f"REJECTED ({result['confidence']:.1f}%)"
            elif result['status'] == 'NO_FACE':
                color = (255, 0, 0)  # Blue
                text = "No Face Detected"
            else:
                color = (128, 128, 128)  # Gray
                text = result['status']

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Draw status bar at top
        status_text = f"Real-Time KYC - Frame: {id(self)}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python realtime_kyc.py <id_photo_path>")
        print("Example: python realtime_kyc.py data/test/id_card.jpg")
        sys.exit(1)

    id_path = sys.argv[1]

    # Check if file exists
    if not Path(id_path).exists():
        print(f"Error: ID photo not found: {id_path}")
        sys.exit(1)

    try:
        # Initialize and run real-time KYC
        kyc = RealTimeKYC(id_path)
        kyc.run()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Error running real-time KYC: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()