"""
CLIP Embedder - Generate embeddings for text and images
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

from .embedding_config import EmbeddingConfig
from utils.print_helper import print_ok, print_error, print_warning, print_info


class CLIPEmbedder:
    """CLIP-based embedder for text and images"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize CLIP embedder
        
        Args:
            model_name: CLIP model name (default from config)
            device: Device to use ('cuda' or 'cpu', default from config)
        """
        self.model_name = model_name or EmbeddingConfig.CLIP_MODEL_NAME
        self.device = device or EmbeddingConfig.DEVICE
        
        print_info(f"Loading CLIP model: {self.model_name}")
        print_info(f"Device: {self.device}")
        
        # Load model and processor
        try:
            self.model = CLIPModel.from_pretrained(
                self.model_name,
                cache_dir=EmbeddingConfig.CACHE_DIR
            ).to(self.device)
            
            self.processor = CLIPProcessor.from_pretrained(
                self.model_name,
                cache_dir=EmbeddingConfig.CACHE_DIR
            )
            
            self.model.eval()  # Set to evaluation mode
            print_ok(f"CLIP model loaded successfully")
            
        except Exception as e:
            print_error(f"Failed to load CLIP model: {e}")
            raise
    
    def embed_text(self, texts: Union[str, List[str]], 
                   normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.empty((0, EmbeddingConfig.TEXT_EMBEDDING_DIM))
        
        try:
            # Tokenize
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=EmbeddingConfig.MAX_TEXT_LENGTH
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Convert to numpy
            embeddings = text_features.cpu().numpy()
            
            # Normalize if requested
            if normalize and EmbeddingConfig.NORMALIZE_EMBEDDINGS:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            print_error(f"Error embedding text: {e}")
            raise
    
    def embed_image(self, images: Union[str, Path, Image.Image, List], 
                    normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for image(s)
        
        Args:
            images: Single image (path/PIL) or list of images
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (n_images, embedding_dim)
        """
        # Handle single image
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]
        
        if not images:
            return np.empty((0, EmbeddingConfig.IMAGE_EMBEDDING_DIM))
        
        try:
            # Load images if paths
            pil_images = []
            for img in images:
                if isinstance(img, (str, Path)):
                    try:
                        pil_img = Image.open(img).convert("RGB")
                        pil_images.append(pil_img)
                    except Exception as e:
                        print_warning(f"Failed to load image {img}: {e}")
                        # Use blank image as fallback
                        pil_images.append(Image.new('RGB', (224, 224), color='white'))
                elif isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                else:
                    raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Process images
            inputs = self.processor(
                images=pil_images,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Convert to numpy
            embeddings = image_features.cpu().numpy()
            
            # Normalize if requested
            if normalize and EmbeddingConfig.NORMALIZE_EMBEDDINGS:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
            
        except Exception as e:
            print_error(f"Error embedding images: {e}")
            raise
    
    def embed_multimodal(self, text: str, image: Union[str, Path, Image.Image],
                        normalize: bool = True) -> dict:
        """
        Generate embeddings for both text and image
        
        Args:
            text: Text caption/description
            image: Image (path or PIL Image)
            normalize: Whether to normalize embeddings
            
        Returns:
            Dict with 'text_embedding' and 'image_embedding'
        """
        text_emb = self.embed_text(text, normalize=normalize)
        image_emb = self.embed_image(image, normalize=normalize)
        
        return {
            'text_embedding': text_emb[0],  # Single embedding
            'image_embedding': image_emb[0],
            'similarity': float(np.dot(text_emb[0], image_emb[0]))  # Cosine similarity
        }
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        # Ensure normalized
        emb1 = embedding1 / np.linalg.norm(embedding1)
        emb2 = embedding2 / np.linalg.norm(embedding2)
        
        return float(np.dot(emb1, emb2))
    
    def batch_embed_texts(self, texts: List[str], 
                         batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed texts in batches (for large datasets)
        
        Args:
            texts: List of texts
            batch_size: Batch size (default from config)
            
        Returns:
            Stacked embeddings
        """
        batch_size = batch_size or EmbeddingConfig.BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_text(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)
    
    def batch_embed_images(self, image_paths: List[Union[str, Path]], 
                          batch_size: Optional[int] = None) -> np.ndarray:
        """
        Embed images in batches (for large datasets)
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size (default from config)
            
        Returns:
            Stacked embeddings
        """
        batch_size = batch_size or EmbeddingConfig.BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            embeddings = self.embed_image(batch)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


# Convenience functions
def get_embedder(model_name: Optional[str] = None) -> CLIPEmbedder:
    """Get a CLIP embedder instance"""
    return CLIPEmbedder(model_name=model_name)


if __name__ == "__main__":
    # Quick test
    embedder = CLIPEmbedder()
    
    # Test text embedding
    text_emb = embedder.embed_text("A diagram showing neural network architecture")
    print(f"Text embedding shape: {text_emb.shape}")
    
    # Test with dummy image
    dummy_img = Image.new('RGB', (224, 224), color='blue')
    img_emb = embedder.embed_image(dummy_img)
    print(f"Image embedding shape: {img_emb.shape}")
    
    print_ok("CLIP embedder test passed!")