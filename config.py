"""
Configuration settings for KYC Identity Verification System
"""

# Model Configuration
MODEL_CONFIG = {
    "input_shape": (224, 224, 3),
    "embedding_dim": 128,
    "base_model": "resnet50",
    "weights": "imagenet"
}

# Face Detection Configuration
FACE_DETECTION_CONFIG = {
    "detector": "mtcnn",
    "min_face_size": 20,
    "detection_threshold": 0.5
}

# Verification Configuration
VERIFICATION_CONFIG = {
    "distance_threshold": 0.6,  # Euclidean distance threshold
    "metric": "euclidean",  # "euclidean" or "cosine"
    "normalize_embeddings": True
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-4,
    "loss_function": "triplet",  # "triplet" or "contrastive"
    "margin": 0.5,
    "validation_split": 0.2
}

# Data Configuration
DATA_CONFIG = {
    "train_path": "data/train/",
    "test_path": "data/test/",
    "model_save_path": "models/kyc_model.h5",
    "log_path": "logs/"
}

# Device Configuration
DEVICE_CONFIG = {
    "use_gpu": True,
    "gpu_memory_fraction": 0.7
}
