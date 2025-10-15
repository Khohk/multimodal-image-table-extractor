"""
MongoDB Schema Definitions for Multimodal Extractor
"""

from typing import Dict, List, Optional
from datetime import datetime


class MongoDBSchema:
    """Schema definitions for MongoDB collections"""
    
    # Collection names
    PAPERS_COLLECTION = "papers"
    FIGURES_COLLECTION = "figures"
    TABLES_COLLECTION = "tables"
    EMBEDDINGS_COLLECTION = "embeddings"
    
    @staticmethod
    def paper_document(pdf_path: str, metadata: Dict) -> Dict:
        """
        Schema for papers collection
        
        Stores high-level paper metadata
        """
        return {
            "pdf_path": pdf_path,
            "pdf_name": metadata.get("pdf_file", ""),
            "total_pages": metadata.get("total_pages", 0),
            "extraction_date": datetime.utcnow(),
            "figure_count": len(metadata.get("figures", [])),
            "table_count": len(metadata.get("tables", [])),
            "status": "processed",
            "metadata": {
                "extractor_version": "1.0",
                "has_analysis": False,
                "has_embeddings": False
            }
        }
    
    @staticmethod
    def figure_document(figure_meta: Dict, analysis_result: Optional[Dict] = None,
                       embeddings: Optional[Dict] = None, paper_id: Optional[str] = None) -> Dict:
        """
        Schema for figures collection
        
        Stores figure metadata, analysis, and embeddings
        """
        doc = {
            "paper_id": paper_id,  # Reference to papers collection
            "type": "figure",
            "page": figure_meta.get("page"),
            "index": figure_meta.get("index"),
            "caption": figure_meta.get("caption", ""),
            "filename": figure_meta.get("filename", ""),
            "file_path": figure_meta.get("path", ""),
            "bbox": figure_meta.get("bbox", []),
            "extraction_metadata": {
                "extraction_date": datetime.utcnow()
            }
        }
        
        # Add analysis if available
        if analysis_result:
            doc["analysis"] = {
                "summary": analysis_result.get("summary", ""),
                "model": analysis_result.get("model", ""),
                "status": analysis_result.get("status", ""),
                "analyzed_date": datetime.utcnow()
            }
            if "error" in analysis_result:
                doc["analysis"]["error"] = analysis_result["error"]
        
        # Add embeddings if available
        if embeddings:
            doc["embeddings"] = {
                "text_embedding": embeddings.get("text_embedding", []),
                "image_embedding": embeddings.get("image_embedding", []),
                "similarity": embeddings.get("similarity", 0.0),
                "embedding_model": embeddings.get("model", "clip-vit-base-patch32"),
                "embedding_dim": len(embeddings.get("text_embedding", [])),
                "created_date": datetime.utcnow()
            }
        
        return doc
    
    @staticmethod
    def table_document(table_meta: Dict, analysis_result: Optional[Dict] = None,
                      embeddings: Optional[Dict] = None, paper_id: Optional[str] = None) -> Dict:
        """
        Schema for tables collection
        
        Stores table metadata, analysis, and embeddings
        """
        doc = {
            "paper_id": paper_id,
            "type": "table",
            "page": table_meta.get("page"),
            "index": table_meta.get("index"),
            "caption": table_meta.get("caption", ""),
            "filename": table_meta.get("filename", ""),
            "file_path": table_meta.get("path", ""),
            "bbox": table_meta.get("bbox", []),
            "structure": {
                "rows": table_meta.get("rows", 0),
                "columns": table_meta.get("columns", 0),
                "has_hierarchical_headers": table_meta.get("has_hierarchical_headers", False),
                "extraction_method": table_meta.get("extraction_method", "unknown")
            },
            "extraction_metadata": {
                "extraction_date": datetime.utcnow()
            }
        }
        
        # Add analysis if available
        if analysis_result:
            doc["analysis"] = {
                "summary": analysis_result.get("summary", ""),
                "summary_json": analysis_result.get("summary_json"),
                "summary_mode": analysis_result.get("summary_mode", "text"),
                "model": analysis_result.get("model", ""),
                "status": analysis_result.get("status", ""),
                "stats": analysis_result.get("stats", {}),
                "analyzed_date": datetime.utcnow()
            }
            if "error" in analysis_result:
                doc["analysis"]["error"] = analysis_result["error"]
        
        # Add embeddings if available (text only for tables)
        if embeddings:
            doc["embeddings"] = {
                "text_embedding": embeddings.get("text_embedding", []),
                "embedding_model": embeddings.get("model", "clip-vit-base-patch32"),
                "embedding_dim": len(embeddings.get("text_embedding", [])),
                "created_date": datetime.utcnow()
            }
        
        return doc
    
    @staticmethod
    def get_indexes() -> Dict[str, List[Dict]]:
        """
        Define indexes for collections
        
        Returns:
            Dict mapping collection names to index specifications
        """
        return {
            "papers": [
                {"keys": [("pdf_name", 1)], "unique": True},
                {"keys": [("extraction_date", -1)]},
            ],
            "figures": [
                {"keys": [("paper_id", 1)]},
                {"keys": [("page", 1), ("index", 1)]},
                {"keys": [("caption", "text")]},  # Text search
            ],
            "tables": [
                {"keys": [("paper_id", 1)]},
                {"keys": [("page", 1), ("index", 1)]},
                {"keys": [("caption", "text")]},  # Text search
            ],
        }
    
    @staticmethod
    def validate_document(doc: Dict, doc_type: str) -> bool:
        """
        Validate document structure
        
        Args:
            doc: Document to validate
            doc_type: Type ('paper', 'figure', 'table')
            
        Returns:
            True if valid
        """
        required_fields = {
            "paper": ["pdf_path", "pdf_name"],
            "figure": ["type", "page", "caption", "file_path"],
            "table": ["type", "page", "caption", "file_path"]
        }
        
        if doc_type not in required_fields:
            return False
        
        return all(field in doc for field in required_fields[doc_type])


# Example document structures for reference
EXAMPLE_FIGURE_DOC = {
    "paper_id": "507f1f77bcf86cd799439011",
    "type": "figure",
    "page": 3,
    "index": 1,
    "caption": "Figure 1: CLIP architecture overview",
    "filename": "page3_fig1.png",
    "file_path": "extracted_content/figures/page3_fig1.png",
    "bbox": [100, 200, 500, 600],
    "extraction_metadata": {
        "extraction_date": "2025-10-14T10:30:00Z"
    },
    "analysis": {
        "summary": "Shows CLIP dual-encoder architecture...",
        "model": "gemini-2.0-flash-exp",
        "status": "success",
        "analyzed_date": "2025-10-14T10:35:00Z"
    },
    "embeddings": {
        "text_embedding": [0.123, -0.456, ...],  # 512-dim
        "image_embedding": [0.789, -0.234, ...],  # 512-dim
        "similarity": 0.87,
        "embedding_model": "clip-vit-base-patch32",
        "embedding_dim": 512,
        "created_date": "2025-10-14T10:40:00Z"
    }
}

EXAMPLE_TABLE_DOC = {
    "paper_id": "507f1f77bcf86cd799439011",
    "type": "table",
    "page": 5,
    "index": 1,
    "caption": "Table 1: Performance comparison",
    "filename": "page5_table1.json",
    "file_path": "extracted_content/tables/page5_table1.json",
    "bbox": [100, 200, 500, 400],
    "structure": {
        "rows": 8,
        "columns": 5,
        "has_hierarchical_headers": True,
        "extraction_method": "camelot_lattice"
    },
    "extraction_metadata": {
        "extraction_date": "2025-10-14T10:30:00Z"
    },
    "analysis": {
        "summary": "Compares accuracy across 3 models...",
        "summary_json": {
            "metrics": ["Accuracy", "F1"],
            "observations": ["CLIP performs best"],
            "insight": "Trade-off between speed and accuracy"
        },
        "summary_mode": "json",
        "model": "gemini-2.0-flash-exp",
        "status": "success",
        "stats": {
            "rows": 8,
            "columns": 5
        },
        "analyzed_date": "2025-10-14T10:35:00Z"
    },
    "embeddings": {
        "text_embedding": [0.123, -0.456, ...],  # 512-dim
        "embedding_model": "clip-vit-base-patch32",
        "embedding_dim": 512,
        "created_date": "2025-10-14T10:40:00Z"
    }
}