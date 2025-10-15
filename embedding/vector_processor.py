"""
Vector Processor - Batch processing for embeddings
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

from .clip_embedder import CLIPEmbedder
from .embedding_config import EmbeddingConfig
from utils.print_helper import print_ok, print_error, print_warning, print_info


class VectorProcessor:
    """Process and generate embeddings in batches"""
    
    def __init__(self, embedder: Optional[CLIPEmbedder] = None):
        """
        Initialize vector processor
        
        Args:
            embedder: CLIP embedder instance (creates new if None)
        """
        self.embedder = embedder or CLIPEmbedder()
    
    def process_figures(self, figures_meta: List[Dict], 
                       analysis_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Process figures and generate embeddings
        
        Args:
            figures_meta: List of figure metadata
            analysis_results: List of AI analysis results (optional)
            
        Returns:
            List of dicts with embeddings added
        """
        if not figures_meta:
            print_warning("No figures to process")
            return []
        
        # Match analysis results
        analysis_map = {}
        if analysis_results:
            for result in analysis_results:
                analysis_map[result['path']] = result
        
        processed = []
        print_info(f"Processing {len(figures_meta)} figures...")
        
        for fig_meta in tqdm(figures_meta, desc="Embedding figures"):
            try:
                # Get caption (from metadata or analysis)
                caption = fig_meta.get('caption', '')
                image_path = fig_meta.get('path', '')
                
                # Get analysis if available
                analysis = analysis_map.get(image_path)
                
                # Create text for embedding (caption + summary)
                text_for_embedding = caption
                if analysis and analysis.get('status') == 'success':
                    summary = analysis.get('summary', '')
                    if summary:
                        text_for_embedding = f"{caption}\n\n{summary}"
                
                # Generate embeddings
                multimodal_emb = self.embedder.embed_multimodal(
                    text=text_for_embedding,
                    image=image_path
                )
                
                # Prepare result
                result = {
                    **fig_meta,
                    'embeddings': {
                        'text_embedding': multimodal_emb['text_embedding'].tolist(),
                        'image_embedding': multimodal_emb['image_embedding'].tolist(),
                        'similarity': multimodal_emb['similarity'],
                        'model': self.embedder.model_name
                    }
                }
                
                # Add analysis if available
                if analysis:
                    result['analysis'] = analysis
                
                processed.append(result)
                
            except Exception as e:
                print_error(f"Error processing figure {fig_meta.get('filename', 'unknown')}: {e}")
                # Add without embeddings
                processed.append({**fig_meta, 'embeddings': None, 'error': str(e)})
        
        print_ok(f"Processed {len(processed)} figures")
        return processed
    
    def process_tables(self, tables_meta: List[Dict],
                      analysis_results: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Process tables and generate text embeddings
        
        Args:
            tables_meta: List of table metadata
            analysis_results: List of AI analysis results (optional)
            
        Returns:
            List of dicts with embeddings added
        """
        if not tables_meta:
            print_warning("No tables to process")
            return []
        
        # Match analysis results
        analysis_map = {}
        if analysis_results:
            for result in analysis_results:
                analysis_map[result['path']] = result
        
        processed = []
        print_info(f"Processing {len(tables_meta)} tables...")
        
        for table_meta in tqdm(tables_meta, desc="Embedding tables"):
            try:
                # Get caption
                caption = table_meta.get('caption', '')
                
                # Get analysis if available
                analysis = analysis_map.get(table_meta.get('path', ''))
                
                # Create text for embedding
                text_for_embedding = caption
                if analysis and analysis.get('status') == 'success':
                    summary = analysis.get('summary', '')
                    if summary:
                        text_for_embedding = f"{caption}\n\n{summary}"
                    
                    # Add JSON summary if available
                    if analysis.get('summary_json'):
                        json_summary = json.dumps(analysis['summary_json'])
                        text_for_embedding = f"{text_for_embedding}\n\n{json_summary}"
                
                # Generate text embedding only (tables don't have images in this case)
                text_emb = self.embedder.embed_text(text_for_embedding)
                
                # Prepare result
                result = {
                    **table_meta,
                    'embeddings': {
                        'text_embedding': text_emb[0].tolist(),
                        'model': self.embedder.model_name
                    }
                }
                
                # Add analysis if available
                if analysis:
                    result['analysis'] = analysis
                
                processed.append(result)
                
            except Exception as e:
                print_error(f"Error processing table {table_meta.get('filename', 'unknown')}: {e}")
                processed.append({**table_meta, 'embeddings': None, 'error': str(e)})
        
        print_ok(f"Processed {len(processed)} tables")
        return processed
    
    def process_from_metadata(self, metadata_path: str,
                             analysis_results_path: Optional[str] = None) -> Dict:
        """
        Process from metadata.json file
        
        Args:
            metadata_path: Path to metadata.json
            analysis_results_path: Path to analysis_results.json (optional)
            
        Returns:
            Dict with processed figures and tables
        """
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load analysis results if available
        analysis_results = []
        if analysis_results_path and Path(analysis_results_path).exists():
            with open(analysis_results_path, 'r', encoding='utf-8') as f:
                analysis_results = json.load(f)
        
        # Process figures and tables
        figures = self.process_figures(
            metadata.get('figures', []),
            analysis_results
        )
        
        tables = self.process_tables(
            metadata.get('tables', []),
            analysis_results
        )
        
        return {
            'pdf_file': metadata.get('pdf_file', ''),
            'total_pages': metadata.get('total_pages', 0),
            'figures': figures,
            'tables': tables,
            'embedding_model': self.embedder.model_name,
            'embedding_dim': EmbeddingConfig.TEXT_EMBEDDING_DIM
        }
    
    def save_processed_data(self, processed_data: Dict, output_path: str):
        """Save processed data with embeddings to JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print_ok(f"Saved processed data to: {output_path}")


if __name__ == "__main__":
    # Quick test
    processor = VectorProcessor()
    
    # Test with dummy data
    test_figures = [{
        'path': 'test.png',
        'caption': 'Test figure',
        'page': 1,
        'index': 1
    }]
    
    # This would fail without actual image, but shows the interface
    print_info("VectorProcessor initialized successfully")
    print_ok("Ready to process embeddings!")