"""
Configuration for Embedding Pipeline
"""

import os
import torch
from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig:
    """Configuration for CLIP embedding"""
    
    # Model Configuration
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
    # Alternative models:
    # - "openai/clip-vit-large-patch14" (better quality, slower)
    # - "openai/clip-vit-base-patch16" (balanced)
    
    # Device Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Embedding Dimensions
    TEXT_EMBEDDING_DIM = 512   # CLIP text encoder output
    IMAGE_EMBEDDING_DIM = 512  # CLIP image encoder output
    
    # Processing Configuration
    BATCH_SIZE = 16  # Batch size for embedding generation
    MAX_TEXT_LENGTH = 77  # CLIP's max token length
    IMAGE_SIZE = 224  # CLIP's expected image size
    
    # Normalization (recommended for cosine similarity)
    NORMALIZE_EMBEDDINGS = True
    
    # Cache Configuration
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "models")
    
    @classmethod
    def validate(cls):
        """Validate embedding configuration"""
        if cls.DEVICE == "cuda":
            if not torch.cuda.is_available():
                print("[WARNING] CUDA not available, falling back to CPU")
                cls.DEVICE = "cpu"
            else:
                print(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] Using CPU for embeddings (consider GPU for faster processing)")
        
        # Create cache directory
        os.makedirs(cls.CACHE_DIR, exist_ok=True)
        print(f"[OK] Model cache directory: {cls.CACHE_DIR}")


# Validate on import
EmbeddingConfig.validate()