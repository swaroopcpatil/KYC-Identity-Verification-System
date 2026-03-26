"""
Siamese CNN model for face verification using ResNet50
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import Loss
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletLoss(Loss):
    """Custom Triplet Loss for face embeddings"""
    
    def __init__(self, margin=0.5, **kwargs):
        """
        Initialize Triplet Loss
        
        Args:
            margin: Margin for triplet loss (default: 0.5)
        """
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """
        Calculate triplet loss
        
        Args:
            y_true: True labels
            y_pred: Predicted embeddings [anchor, positive, negative]
            
        Returns:
            Loss value
        """
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        
        # Euclidean distance
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        
        # Triplet loss
        loss = tf.maximum(pos_dist - neg_dist + self.margin, 0.0)
        return tf.reduce_mean(loss)


class ContrastiveLoss(Loss):
    """Custom Contrastive Loss for face verification"""
    
    def __init__(self, margin=1.0, **kwargs):
        """
        Initialize Contrastive Loss
        
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """
        Calculate contrastive loss
        
        Args:
            y_true: Binary labels (1 = same person, 0 = different)
            y_pred: Euclidean distance between embeddings
            
        Returns:
            Loss value
        """
        y_true = tf.cast(y_true, tf.float32)
        # Contrastive loss formula
        loss = y_true * tf.square(y_pred) + (1 - y_true) * tf.square(
            tf.maximum(self.margin - y_pred, 0.0)
        )
        return tf.reduce_mean(loss)


class SiameseNetwork:
    """Siamese CNN Network for face verification"""
    
    def __init__(self, input_shape=(224, 224, 3), embedding_dim=128):
        """
        Initialize Siamese Network
        
        Args:
            input_shape: Input image shape
            embedding_dim: Dimension of face embedding
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = None
        self.embedding_model = None
        
        logger.info(f"Initialized SiameseNetwork with embedding_dim={embedding_dim}")
    
    def build_base_network(self) -> Model:
        """
        Build base network using ResNet50
        
        Returns:
            Base model for feature extraction
        """
        # Load pre-trained ResNet50
        base_model = ResNet50(
            input_shape=self.input_shape,
            weights='imagenet',
            include_top=False
        )
        
        # Freeze base model weights for transfer learning
        for layer in base_model.layers:
            layer.trainable = False
        
        # Create feature extraction model
        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Embedding layer with L2 normalization
        embeddings = layers.Dense(self.embedding_dim, kernel_regularizer='l2', name='embedding')(x)
        embeddings = layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1),
            name='l2_norm'
        )(embeddings)
        
        return Model(inputs, embeddings, name='base_network')
    
    def build_siamese_model(self) -> Model:
        """
        Build complete Siamese network
        
        Returns:
            Siamese model taking two inputs
        """
        base_network = self.build_base_network()
        
        # Two inputs for image pairs
        input_a = keras.Input(shape=self.input_shape, name='input_a')
        input_b = keras.Input(shape=self.input_shape, name='input_b')
        
        # Process both inputs through same network
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        # Calculate Euclidean distance
        distance = layers.Lambda(
            lambda x: tf.sqrt(tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True)),
            output_shape=(1,),
            name='distance'
        )([embedding_a, embedding_b])
        
        # Binary classification (same/different person)
        outputs = layers.Dense(1, activation='sigmoid', name='prediction')(distance)
        
        siamese_model = Model(
            inputs=[input_a, input_b],
            outputs=outputs,
            name='siamese_network'
        )
        
        self.embedding_model = base_network
        self.model = siamese_model
        
        logger.info("Siamese model built successfully")
        return siamese_model
    
    def compile_model(self, learning_rate=1e-4, loss='binary_crossentropy'):
        """
        Compile the Siamese model
        
        Args:
            learning_rate: Learning rate for optimizer
            loss: Loss function ('binary_crossentropy', 'triplet', or 'contrastive')
        """
        if self.model is None:
            self.build_siamese_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        if loss == 'triplet':
            loss_fn = TripletLoss(margin=0.5)
        elif loss == 'contrastive':
            loss_fn = ContrastiveLoss(margin=1.0)
        else:
            loss_fn = loss
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info(f"Model compiled with {loss} loss")
    
    def get_embedding_model(self) -> Model:
        """
        Get the embedding extraction model
        
        Returns:
            Model that outputs face embeddings
        """
        if self.embedding_model is None:
            self.build_siamese_model()
        return self.embedding_model
    
    def get_siamese_model(self) -> Model:
        """
        Get the complete Siamese model
        
        Returns:
            Complete Siamese network model
        """
        if self.model is None:
            self.build_siamese_model()
        return self.model
    
    def print_model_summary(self):
        """Print model architecture summary"""
        if self.model is not None:
            self.model.summary()
        if self.embedding_model is not None:
            self.embedding_model.summary()


def create_model(embedding_dim=128, input_shape=(224, 224, 3)) -> Model:
    """
    Factory function to create Siamese network
    
    Args:
        embedding_dim: Dimension of embeddings
        input_shape: Input image shape
        
    Returns:
        Compiled Siamese model
    """
    siamese = SiameseNetwork(input_shape=input_shape, embedding_dim=embedding_dim)
    siamese.build_siamese_model()
    siamese.compile_model(learning_rate=1e-4, loss='binary_crossentropy')
    return siamese


if __name__ == "__main__":
    logger.info("Building model for testing...")
    siamese = SiameseNetwork()
    siamese.build_siamese_model()
    siamese.compile_model()
    siamese.print_model_summary()
