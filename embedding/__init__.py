"""Embedding module - CLIP-based embeddings"""

from .clip_embedder import CLIPEmbedder, get_embedder
from .embedding_config import EmbeddingConfig
from .vector_processor import VectorProcessor

__all__ = [
    'CLIPEmbedder',
    'get_embedder',
    'EmbeddingConfig',
    'VectorProcessor'
]