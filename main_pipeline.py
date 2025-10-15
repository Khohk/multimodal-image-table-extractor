"""
Main Pipeline - Complete Multimodal PDF Extraction to MongoDB
Integrates: Detection → Analysis → Embedding → Storage
Enhanced: Support single sample or batch processing
"""

# Fix Windows encoding
import sys
import os
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

import asyncio
import json
from pathlib import Path
from typing import Dict, Optional, List

from analysis import AIAnalyzer
from embedding import VectorProcessor
from storage import MongoDBHandler
from utils.print_helper import (
    print_separator, print_ok, print_error, 
    print_info, print_warning
)


class MultimodalPipeline:
    """Complete pipeline for multimodal PDF processing"""
    
    def __init__(self, mongodb_uri: Optional[str] = None):
        """
        Initialize pipeline
        
        Args:
            mongodb_uri: MongoDB connection URI (optional)
        """
        self.analyzer = None
        self.processor = None
        self.db_handler = None
        self.mongodb_uri = mongodb_uri
        
        print_separator("MULTIMODAL PDF EXTRACTION PIPELINE")
        print_info("Initializing components...")
    
    def find_all_samples(self, base_dir: str = "extracted_content") -> List[Path]:
        """
        Find all sample directories with metadata.json
        
        Args:
            base_dir: Base directory containing samples
            
        Returns:
            List of paths to metadata.json files
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            print_error(f"Base directory not found: {base_dir}")
            return []
        
        metadata_files = []
        
        # Search for all metadata.json files in subdirectories
        for sample_dir in base_path.iterdir():
            if sample_dir.is_dir():
                metadata_path = sample_dir / "metadata.json"
                if metadata_path.exists():
                    metadata_files.append(metadata_path)
        
        return sorted(metadata_files)
    
    async def process_pdf(self, metadata_path: str, 
                         skip_analysis: bool = False,
                         skip_embedding: bool = False,
                         skip_storage: bool = False) -> Dict:
        """
        Process a single PDF through the complete pipeline
        
        Args:
            metadata_path: Path to metadata.json from extraction
            skip_analysis: Skip AI analysis step
            skip_embedding: Skip embedding generation
            skip_storage: Skip MongoDB storage
            
        Returns:
            Dict with processing results
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        pdf_name = metadata.get('pdf_file', 'unknown.pdf')
        sample_name = metadata_path.parent.name
        
        print_separator(f"PROCESSING: {sample_name} ({pdf_name})")
        
        results = {
            'sample': sample_name,
            'pdf_file': pdf_name,
            'metadata_path': str(metadata_path),
            'steps_completed': [],
            'success': True,
            'error': None
        }
        
        try:
            # ==================== STEP 1: AI ANALYSIS ====================
            analysis_results = []
            analysis_path = metadata_path.parent / "analysis_results.json"
            
            if not skip_analysis:
                print_separator("STEP 1: AI Analysis")
                
                if analysis_path.exists():
                    print_info("Found existing analysis results")
                    with open(analysis_path, 'r', encoding='utf-8') as f:
                        analysis_results = json.load(f)
                else:
                    print_info("Running AI analysis...")
                    
                    # Initialize analyzer
                    if not self.analyzer:
                        self.analyzer = AIAnalyzer()
                    
                    # Prepare items for analysis
                    items = []
                    for fig in metadata.get('figures', []):
                        items.append({
                            'path': fig['path'],
                            'type': 'figure',
                            'caption': fig.get('caption', '')
                        })
                    
                    for table in metadata.get('tables', []):
                        items.append({
                            'path': table['path'],
                            'type': 'table',
                            'caption': table.get('caption', ''),
                            'summary_mode': 'json'
                        })
                    
                    # Analyze
                    if items:
                        analysis_results = await self.analyzer.analyze_batch(items)
                        
                        # Save results
                        with open(analysis_path, 'w', encoding='utf-8') as f:
                            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
                        
                        print_ok(f"Analysis complete: {len(analysis_results)} items")
                    else:
                        print_warning("No items to analyze")
                
                results['steps_completed'].append('analysis')
                results['analysis_count'] = len(analysis_results)
            
            # ==================== STEP 2: EMBEDDING ====================
            processed_data = None
            processed_path = metadata_path.parent / "processed_with_embeddings.json"
            
            if not skip_embedding:
                print_separator("STEP 2: Embedding Generation")
                
                if processed_path.exists():
                    print_info("Found existing embeddings")
                    with open(processed_path, 'r', encoding='utf-8') as f:
                        processed_data = json.load(f)
                else:
                    print_info("Generating embeddings...")
                    
                    # Initialize processor
                    if not self.processor:
                        self.processor = VectorProcessor()
                    
                    # Process
                    processed_data = self.processor.process_from_metadata(
                        str(metadata_path),
                        str(analysis_path) if analysis_path.exists() else None
                    )
                    
                    # Save
                    self.processor.save_processed_data(processed_data, str(processed_path))
                    
                    print_ok("Embeddings generated successfully")
                
                results['steps_completed'].append('embedding')
                results['embedded_figures'] = len(processed_data.get('figures', []))
                results['embedded_tables'] = len(processed_data.get('tables', []))
            
            # ==================== STEP 3: MONGODB STORAGE ====================
            if not skip_storage:
                print_separator("STEP 3: MongoDB Storage")
                
                if not processed_data:
                    if processed_path.exists():
                        with open(processed_path, 'r', encoding='utf-8') as f:
                            processed_data = json.load(f)
                    else:
                        print_error("No processed data found. Run embedding step first.")
                        results['success'] = False
                        results['error'] = "No processed data available"
                        return results
                
                # Initialize MongoDB handler
                if not self.db_handler:
                    try:
                        self.db_handler = MongoDBHandler(uri=self.mongodb_uri)
                    except Exception as e:
                        print_error(f"MongoDB connection failed: {e}")
                        results['success'] = False
                        results['error'] = f"MongoDB error: {e}"
                        return results
                
                # Insert paper
                paper_id = self.db_handler.insert_paper(
                    pdf_path=metadata.get('pdf_file', ''),
                    metadata=metadata
                )
                
                results['paper_id'] = paper_id
                print_info(f"Paper ID: {paper_id}")
                
                # Insert figures
                if processed_data.get('figures'):
                    print_info(f"Inserting {len(processed_data['figures'])} figures...")
                    for fig in processed_data['figures']:
                        self.db_handler.insert_figure(
                            figure_meta=fig,
                            analysis_result=fig.get('analysis'),
                            embeddings=fig.get('embeddings'),
                            paper_id=paper_id
                        )
                    print_ok("Figures inserted")
                
                # Insert tables
                if processed_data.get('tables'):
                    print_info(f"Inserting {len(processed_data['tables'])} tables...")
                    for table in processed_data['tables']:
                        self.db_handler.insert_table(
                            table_meta=table,
                            analysis_result=table.get('analysis'),
                            embeddings=table.get('embeddings'),
                            paper_id=paper_id
                        )
                    print_ok("Tables inserted")
                
                # Update paper status
                self.db_handler.update_paper_status(paper_id, {
                    'has_analysis': not skip_analysis,
                    'has_embeddings': not skip_embedding,
                    'processing_complete': True
                })
                
                results['steps_completed'].append('storage')
            
            # ==================== SUMMARY ====================
            print_separator(f"PROCESSING COMPLETE: {sample_name}")
            print_info(f"PDF: {pdf_name}")
            print_info(f"Steps completed: {', '.join(results['steps_completed'])}")
            
            if 'paper_id' in results:
                print_ok(f"Stored in MongoDB with ID: {results['paper_id']}")
        
        except Exception as e:
            print_error(f"Error processing {sample_name}: {e}")
            results['success'] = False
            results['error'] = str(e)
            import traceback
            traceback.print_exc()
        
        return results
    
    async def process_all_samples(self,
                                  base_dir: str = "extracted_content",
                                  skip_analysis: bool = False,
                                  skip_embedding: bool = False,
                                  skip_storage: bool = False) -> List[Dict]:
        """
        Process all samples in the base directory
        
        Args:
            base_dir: Base directory containing samples
            skip_analysis: Skip AI analysis step
            skip_embedding: Skip embedding generation
            skip_storage: Skip MongoDB storage
            
        Returns:
            List of processing results for each sample
        """
        metadata_files = self.find_all_samples(base_dir)
        
        if not metadata_files:
            print_error(f"No samples found in {base_dir}")
            return []
        
        print_separator("BATCH PROCESSING")
        print_info(f"Found {len(metadata_files)} samples to process")
        
        all_results = []
        
        for i, metadata_path in enumerate(metadata_files, 1):
            print_separator(f"SAMPLE {i}/{len(metadata_files)}")
            
            result = await self.process_pdf(
                metadata_path=str(metadata_path),
                skip_analysis=skip_analysis,
                skip_embedding=skip_embedding,
                skip_storage=skip_storage
            )
            
            all_results.append(result)
        
        # Print summary
        print_separator("BATCH PROCESSING SUMMARY")
        successful = sum(1 for r in all_results if r['success'])
        failed = len(all_results) - successful
        
        print_info(f"Total samples: {len(all_results)}")
        print_ok(f"Successful: {successful}")
        if failed > 0:
            print_error(f"Failed: {failed}")
        
        # Show statistics if MongoDB was used
        if not skip_storage and self.db_handler:
            self.db_handler.print_statistics()
        
        return all_results
    
    def close(self):
        """Clean up resources"""
        if self.db_handler:
            self.db_handler.close()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal PDF Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a specific sample
  python pipeline.py extracted_content/sample_A/metadata.json
  
  # Process all samples in extracted_content
  python pipeline.py --all
  
  # Process all samples, skip analysis (use existing)
  python pipeline.py --all --skip-analysis
  
  # Process specific sample without storage
  python pipeline.py extracted_content/sample_B/metadata.json --skip-storage
        """
    )
    
    parser.add_argument(
        'metadata',
        nargs='?',
        help='Path to metadata.json from extraction (for single sample processing)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all samples in extracted_content directory'
    )
    parser.add_argument(
        '--base-dir',
        default='extracted_content',
        help='Base directory containing samples (default: extracted_content)'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Skip AI analysis step'
    )
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='Skip embedding generation'
    )
    parser.add_argument(
        '--skip-storage',
        action='store_true',
        help='Skip MongoDB storage'
    )
    parser.add_argument(
        '--mongodb-uri',
        default=None,
        help='MongoDB connection URI'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.metadata:
        parser.error("Either provide a metadata path or use --all flag")
    
    # Run pipeline
    pipeline = MultimodalPipeline(mongodb_uri=args.mongodb_uri)
    
    try:
        if args.all:
            # Process all samples
            results = await pipeline.process_all_samples(
                base_dir=args.base_dir,
                skip_analysis=args.skip_analysis,
                skip_embedding=args.skip_embedding,
                skip_storage=args.skip_storage
            )
            
            # Save batch results
            results_file = Path(args.base_dir) / "batch_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print_separator("BATCH RESULTS")
            print_info(f"Results saved to: {results_file}")
            
        else:
            # Process single sample
            results = await pipeline.process_pdf(
                metadata_path=args.metadata,
                skip_analysis=args.skip_analysis,
                skip_embedding=args.skip_embedding,
                skip_storage=args.skip_storage
            )
            
            print_separator("RESULTS")
            print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())