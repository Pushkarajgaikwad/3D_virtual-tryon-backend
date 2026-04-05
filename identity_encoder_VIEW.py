"""
Identity Encoder for Virtual Try-On Backend.

Handles:
- Face embedding extraction (512-dimensional)
- ArcFace-compatible identity representation
- Lightweight pretrained model loading
- Face preprocessing and normalization

Key features:
- 512-D face embeddings for identity matching
- Normalized embeddings for cosine similarity
- CPU-optimized inference
- Graceful handling of missing models
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class IdentityEncoder(nn.Module):
    """
    Lightweight face identity encoder.
    
    Extracts 512-dimensional embeddings from face images using a
    simple ResNet-18 backbone + identity projection head.
    
    Compatible with ArcFace loss for identity preservation.
    
    Architecture:
    - Backbone: ResNet-18 (pretrained on ImageNet)
    - Output layer: 512-D embedding space
    - Preprocessing: Normalize to ImageNet stats
    
    Outputs:
    - 512-D face embedding (unit-norm for cosine similarity)
    """
    
    def __init__(self, embedding_dim: int = 512):
        """
        Initialize IdentityEncoder.
        
        Args:
            embedding_dim: Output embedding dimension. Default: 512.
        
        Raises:
            RuntimeError: If pretrained model cannot be loaded.
        """
        super(IdentityEncoder, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        try:
            # Load pretrained ResNet-18
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=True)
            
            # Remove classification head, keep features
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
            
            # Get feature dimension from backbone
            feature_dim = 512  # ResNet-18 outputs 512 features
            
            # Add projection head to get embedding_dim
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim)
            )
            
            logger.info(f"IdentityEncoder initialized: ResNet-18 → {embedding_dim}D embedding")
        
        except Exception as e:
            logger.error(f"Failed to initialize IdentityEncoder: {e}")
            raise RuntimeError(f"Cannot initialize IdentityEncoder: {e}")
    
    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract identity embedding from image tensor.
        
        Args:
            image_tensor: Input tensor (B, 3, H, W) where H=W=112.
                         Values in [0, 1] (normalized by ImageNet stats).
        
        Returns:
            Embedding tensor (B, embedding_dim) with unit norm.
        
        Raises:
            ValueError: If input shape invalid.
        """
        if image_tensor.ndim != 4 or image_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected (B, 3, H, W) tensor, got {image_tensor.shape}"
            )
        
        # Extract features
        features = self.backbone(image_tensor)  # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        
        # Project to embedding space
        embedding = self.projection(features)  # (B, embedding_dim)
        
        # Normalize to unit norm (for cosine similarity)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        
        return embedding
    
    def extract_identity(self, face_image: Image.Image) -> np.ndarray:
        """
        Extract identity embedding from PIL face image.
        
        Workflow:
        1. Preprocess image to 112x112
        2. Normalize using ImageNet statistics
        3. Convert to tensor
        4. Forward pass through encoder
        5. Return as numpy array
        
        Args:
            face_image: PIL Image of aligned face.
        
        Returns:
            Embedding as (512,) numpy array with unit norm.
        
        Raises:
            ValueError: If image cannot be processed.
        """
        try:
            # Preprocessing pipeline
            preprocess = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet statistics
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # Process image
            if isinstance(face_image, Image.Image):
                if face_image.mode != 'RGB':
                    face_image = face_image.convert('RGB')
                image_tensor = preprocess(face_image)
            else:
                raise ValueError(f"Expected PIL Image, got {type(face_image)}")
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)  # (1, 3, 112, 112)
            
            # Move to CPU (explicit)
            image_tensor = image_tensor.to('cpu')
            
            # Extract embedding without gradients
            with torch.no_grad():
                embedding = self.forward(image_tensor)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().squeeze()  # (512,)
            
            logger.debug(f"Extracted identity embedding: shape {embedding_np.shape}, norm {np.linalg.norm(embedding_np):.4f}")
            
            return embedding_np
        
        except Exception as e:
            logger.error(f"Error extracting identity embedding: {e}")
            raise ValueError(f"Cannot extract identity embedding: {e}")
    
    def extract_identity_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """
        Extract identity embedding from image bytes.
        
        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.).
        
        Returns:
            Embedding as (512,) numpy array.
        
        Raises:
            ValueError: If image cannot be decoded or processed.
        """
        try:
            # Decode image
            image = Image.open(BytesIO(image_bytes))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract embedding
            embedding = self.extract_identity(image)
            
            return embedding
        
        except Exception as e:
            logger.error(f"Error extracting embedding from bytes: {e}")
            raise ValueError(f"Cannot extract embedding from bytes: {e}")


# ============================================================================
# Singleton instance for global use
# ============================================================================

_identity_encoder_instance: Optional[IdentityEncoder] = None


def get_identity_encoder(embedding_dim: int = 512) -> IdentityEncoder:
    """
    Get or create singleton IdentityEncoder instance.
    
    Args:
        embedding_dim: Embedding dimension. Default: 512.
    
    Returns:
        IdentityEncoder instance (model in eval mode).
    """
    global _identity_encoder_instance
    
    if _identity_encoder_instance is None:
        logger.info("Creating IdentityEncoder singleton...")
        _identity_encoder_instance = IdentityEncoder(embedding_dim=embedding_dim)
        _identity_encoder_instance.eval()
        _identity_encoder_instance.to('cpu')
    
    return _identity_encoder_instance


# ============================================================================
# Example Usage and Smoke Test
# ============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("IdentityEncoder Smoke Test")
    print("="*70 + "\n")
    
    try:
        # Initialize
        print("[1/4] Initializing IdentityEncoder...")
        encoder = IdentityEncoder(embedding_dim=512)
        encoder.eval()
        print("      ✓ Initialized\n")
        
        # Create dummy face image
        print("[2/4] Creating dummy face image (112x112)...")
        dummy_face = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
        dummy_pil = Image.fromarray(dummy_face, mode='RGB')
        print(f"      Face image: {dummy_pil.size}\n")
        
        # Extract embedding from PIL image
        print("[3/4] Extracting embedding from PIL image...")
        embedding = encoder.extract_identity(dummy_pil)
        print(f"      Embedding shape: {embedding.shape}")
        print(f"      Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"      Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]\n")
        
        # Extract embedding from bytes
        print("[4/4] Extracting embedding from image bytes...")
        img_bytes = BytesIO()
        dummy_pil.save(img_bytes, format='PNG')
        img_bytes_data = img_bytes.getvalue()
        embedding_from_bytes = encoder.extract_identity_from_bytes(img_bytes_data)
        print(f"      Embedding shape: {embedding_from_bytes.shape}")
        print(f"      Embedding norm: {np.linalg.norm(embedding_from_bytes):.4f}\n")
        
        print("="*70)
        print("✓ IdentityEncoder smoke test passed!")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
