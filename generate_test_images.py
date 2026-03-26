"""
Generate sample test images for KYC demo
"""

import numpy as np
import cv2
import os
from pathlib import Path

def create_realistic_face_image(width=224, height=224, face_color=(210, 180, 140)):
    """Create a more realistic face-like image that Haar Cascade can detect"""
    # Create a blank image with light background
    image = np.ones((height, width, 3), dtype=np.uint8) * 200

    center_x, center_y = width // 2, height // 2
    face_radius = min(width, height) // 3

    # Draw face oval (more realistic shape)
    cv2.ellipse(image, (center_x, center_y), (face_radius, int(face_radius * 1.2)),
                0, 0, 360, face_color, -1)

    # Draw eyes (black circles)
    eye_y = center_y - face_radius // 3
    eye_radius = face_radius // 8
    cv2.circle(image, (center_x - face_radius // 3, eye_y), eye_radius, (0, 0, 0), -1)
    cv2.circle(image, (center_x + face_radius // 3, eye_y), eye_radius, (0, 0, 0), -1)

    # Draw nose (small triangle)
    nose_points = np.array([
        [center_x, center_y - face_radius // 6],
        [center_x - face_radius // 12, center_y + face_radius // 6],
        [center_x + face_radius // 12, center_y + face_radius // 6]
    ], np.int32)
    cv2.fillPoly(image, [nose_points], (180, 150, 120))

    # Draw mouth (simple arc)
    mouth_y = center_y + face_radius // 2
    cv2.ellipse(image, (center_x, mouth_y), (face_radius // 4, face_radius // 8),
                0, 0, 180, (150, 50, 50), 2)

    return image

def create_sample_id_card(face_image, width=400, height=250):
    """Create a sample ID card with embedded face"""
    # Create ID card background
    id_card = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light gray

    # Add border
    cv2.rectangle(id_card, (10, 10), (width-10, height-10), (0, 0, 0), 2)

    # Resize face for ID card - make it larger for better detection
    face_resized = cv2.resize(face_image, (120, 150))  # Larger face

    # Place face on ID card
    id_card[20:170, 20:140] = face_resized  # Position for better detection

    # Add text
    cv2.putText(id_card, "SAMPLE ID CARD", (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(id_card, "Name: John Doe", (160, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(id_card, "ID: 123456789", (160, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(id_card, "DOB: 01/01/1990", (160, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return id_card

def main():
    """Generate sample images for KYC testing"""
    test_dir = Path("data/test")
    test_dir.mkdir(exist_ok=True)

    print("Generating sample test images...")

    # Create two different faces for testing
    face1 = create_realistic_face_image(face_color=(210, 180, 140))  # Person 1
    face2 = create_realistic_face_image(face_color=(190, 160, 130))  # Person 2 (different)

    # Save selfie images
    cv2.imwrite(str(test_dir / "selfie.jpg"), face1)
    cv2.imwrite(str(test_dir / "person2.jpg"), face2)

    # Create ID cards
    id_card1 = create_sample_id_card(face1)
    id_card2 = create_sample_id_card(face2)

    # Save ID card images
    cv2.imwrite(str(test_dir / "id_card.jpg"), id_card1)
    cv2.imwrite(str(test_dir / "id_card_person2.jpg"), id_card2)

    print("✓ Generated sample images:")
    print(f"  - {test_dir / 'selfie.jpg'} (Person 1 selfie)")
    print(f"  - {test_dir / 'id_card.jpg'} (Person 1 ID card)")
    print(f"  - {test_dir / 'person2.jpg'} (Person 2 selfie)")
    print(f"  - {test_dir / 'id_card_person2.jpg'} (Person 2 ID card)")
    print("\nThese images can be used to test the KYC system!")

if __name__ == "__main__":
    main()