"""
MongoDB Handler - Store and retrieve multimodal data
"""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from bson.objectid import ObjectId
from dotenv import load_dotenv

from .schema import MongoDBSchema
from utils.print_helper import print_ok, print_error, print_warning, print_info

load_dotenv()


class MongoDBHandler:
    """Handler for MongoDB operations"""
    
    def __init__(self, uri: Optional[str] = None, db_name: str = "multimodal_rnd"):
        """
        Initialize MongoDB handler
        
        Args:
            uri: MongoDB connection URI (default from env)
            db_name: Database name
        """
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.db_name = db_name
        
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.server_info()
            print_ok(f"Connected to MongoDB: {self.db_name}")
            
        except ConnectionFailure as e:
            print_error(f"Failed to connect to MongoDB: {e}")
            print_warning("Make sure MongoDB is running!")
            raise
        
        self.db = self.client[self.db_name]
        
        # Collections
        self.papers = self.db[MongoDBSchema.PAPERS_COLLECTION]
        self.figures = self.db[MongoDBSchema.FIGURES_COLLECTION]
        self.tables = self.db[MongoDBSchema.TABLES_COLLECTION]
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create indexes for collections"""
        try:
            indexes = MongoDBSchema.get_indexes()
            
            # Papers indexes
            for idx in indexes.get("papers", []):
                self.papers.create_index(idx["keys"], unique=idx.get("unique", False))
            
            # Figures indexes
            for idx in indexes.get("figures", []):
                self.figures.create_index(idx["keys"])
            
            # Tables indexes
            for idx in indexes.get("tables", []):
                self.tables.create_index(idx["keys"])
            
            print_ok("Database indexes created")
            
        except Exception as e:
            print_warning(f"Error creating indexes: {e}")
    
    # ==================== PAPER OPERATIONS ====================
    
    def insert_paper(self, pdf_path: str, metadata: Dict) -> str:
        """
        Insert paper document
        
        Args:
            pdf_path: Path to PDF file
            metadata: Extraction metadata
            
        Returns:
            Inserted document ID
        """
        try:
            doc = MongoDBSchema.paper_document(pdf_path, metadata)
            result = self.papers.insert_one(doc)
            print_ok(f"Inserted paper: {doc['pdf_name']}")
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            print_warning(f"Paper already exists: {pdf_path}")
            # Return existing ID
            existing = self.papers.find_one({"pdf_name": doc["pdf_name"]})
            return str(existing["_id"])
        except Exception as e:
            print_error(f"Error inserting paper: {e}")
            raise
    
    def get_paper_by_name(self, pdf_name: str) -> Optional[Dict]:
        """Get paper by PDF name"""
        return self.papers.find_one({"pdf_name": pdf_name})
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """Get paper by ID"""
        try:
            return self.papers.find_one({"_id": ObjectId(paper_id)})
        except Exception:
            return None
    
    def update_paper_status(self, paper_id: str, status_updates: Dict):
        """Update paper metadata"""
        self.papers.update_one(
            {"_id": ObjectId(paper_id)},
            {"$set": {"metadata": status_updates}}
        )
    
    # ==================== FIGURE OPERATIONS ====================
    
    def insert_figure(self, figure_meta: Dict, analysis_result: Optional[Dict] = None,
                     embeddings: Optional[Dict] = None, paper_id: Optional[str] = None) -> str:
        """
        Insert figure document
        
        Args:
            figure_meta: Figure metadata from extraction
            analysis_result: AI analysis results
            embeddings: CLIP embeddings
            paper_id: Reference to paper
            
        Returns:
            Inserted document ID
        """
        try:
            doc = MongoDBSchema.figure_document(
                figure_meta, analysis_result, embeddings, paper_id
            )
            result = self.figures.insert_one(doc)
            print_info(f"Inserted figure: {doc['filename']}")
            return str(result.inserted_id)
            
        except Exception as e:
            print_error(f"Error inserting figure: {e}")
            raise
    
    def insert_figures_batch(self, figures_data: List[Dict]) -> List[str]:
        """Insert multiple figures"""
        if not figures_data:
            return []
        
        try:
            result = self.figures.insert_many(figures_data)
            print_ok(f"Inserted {len(result.inserted_ids)} figures")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            print_error(f"Error batch inserting figures: {e}")
            raise
    
    def get_figures_by_paper(self, paper_id: str) -> List[Dict]:
        """Get all figures for a paper"""
        return list(self.figures.find({"paper_id": paper_id}))
    
    def update_figure_analysis(self, figure_id: str, analysis_result: Dict):
        """Update figure with analysis results"""
        self.figures.update_one(
            {"_id": ObjectId(figure_id)},
            {"$set": {
                "analysis": {
                    "summary": analysis_result.get("summary", ""),
                    "model": analysis_result.get("model", ""),
                    "status": analysis_result.get("status", ""),
                    "analyzed_date": datetime.utcnow()
                }
            }}
        )
    
    def update_figure_embeddings(self, figure_id: str, embeddings: Dict):
        """Update figure with embeddings"""
        self.figures.update_one(
            {"_id": ObjectId(figure_id)},
            {"$set": {
                "embeddings": {
                    "text_embedding": embeddings.get("text_embedding", []),
                    "image_embedding": embeddings.get("image_embedding", []),
                    "similarity": embeddings.get("similarity", 0.0),
                    "embedding_model": embeddings.get("model", "clip-vit-base-patch32"),
                    "embedding_dim": len(embeddings.get("text_embedding", [])),
                    "created_date": datetime.utcnow()
                }
            }}
        )
    
    # ==================== TABLE OPERATIONS ====================
    
    def insert_table(self, table_meta: Dict, analysis_result: Optional[Dict] = None,
                    embeddings: Optional[Dict] = None, paper_id: Optional[str] = None) -> str:
        """
        Insert table document
        
        Args:
            table_meta: Table metadata from extraction
            analysis_result: AI analysis results
            embeddings: CLIP embeddings
            paper_id: Reference to paper
            
        Returns:
            Inserted document ID
        """
        try:
            doc = MongoDBSchema.table_document(
                table_meta, analysis_result, embeddings, paper_id
            )
            result = self.tables.insert_one(doc)
            print_info(f"Inserted table: {doc['filename']}")
            return str(result.inserted_id)
            
        except Exception as e:
            print_error(f"Error inserting table: {e}")
            raise
    
    def insert_tables_batch(self, tables_data: List[Dict]) -> List[str]:
        """Insert multiple tables"""
        if not tables_data:
            return []
        
        try:
            result = self.tables.insert_many(tables_data)
            print_ok(f"Inserted {len(result.inserted_ids)} tables")
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            print_error(f"Error batch inserting tables: {e}")
            raise
    
    def get_tables_by_paper(self, paper_id: str) -> List[Dict]:
        """Get all tables for a paper"""
        return list(self.tables.find({"paper_id": paper_id}))
    
    def update_table_analysis(self, table_id: str, analysis_result: Dict):
        """Update table with analysis results"""
        self.tables.update_one(
            {"_id": ObjectId(table_id)},
            {"$set": {
                "analysis": {
                    "summary": analysis_result.get("summary", ""),
                    "summary_json": analysis_result.get("summary_json"),
                    "summary_mode": analysis_result.get("summary_mode", "text"),
                    "model": analysis_result.get("model", ""),
                    "status": analysis_result.get("status", ""),
                    "stats": analysis_result.get("stats", {}),
                    "analyzed_date": datetime.utcnow()
                }
            }}
        )
    
    def update_table_embeddings(self, table_id: str, embeddings: Dict):
        """Update table with embeddings"""
        self.tables.update_one(
            {"_id": ObjectId(table_id)},
            {"$set": {
                "embeddings": {
                    "text_embedding": embeddings.get("text_embedding", []),
                    "embedding_model": embeddings.get("model", "clip-vit-base-patch32"),
                    "embedding_dim": len(embeddings.get("text_embedding", [])),
                    "created_date": datetime.utcnow()
                }
            }}
        )
    
    # ==================== SEARCH OPERATIONS ====================
    
    def search_by_caption(self, query: str, element_type: Optional[str] = None) -> List[Dict]:
        """
        Search figures/tables by caption text
        
        Args:
            query: Search query
            element_type: 'figure', 'table', or None (both)
            
        Returns:
            List of matching documents
        """
        search_filter = {"$text": {"$search": query}}
        
        results = []
        if element_type in [None, 'figure']:
            results.extend(list(self.figures.find(search_filter)))
        if element_type in [None, 'table']:
            results.extend(list(self.tables.find(search_filter)))
        
        return results
    
    def get_all_figures(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all figures"""
        query = self.figures.find()
        if limit:
            query = query.limit(limit)
        return list(query)
    
    def get_all_tables(self, limit: Optional[int] = None) -> List[Dict]:
        """Get all tables"""
        query = self.tables.find()
        if limit:
            query = query.limit(limit)
        return list(query)
    
    # ==================== VECTOR SEARCH (for MongoDB Atlas) ====================
    
    def vector_search(self, query_embedding: List[float], 
                     element_type: str = 'figure',
                     top_k: int = 5,
                     embedding_field: str = "embeddings.image_embedding") -> List[Dict]:
        """
        Vector similarity search (requires MongoDB Atlas with vector search)
        
        Args:
            query_embedding: Query vector
            element_type: 'figure' or 'table'
            top_k: Number of results
            embedding_field: Field to search
            
        Returns:
            Top-k similar documents
        """
        collection = self.figures if element_type == 'figure' else self.tables
        
        # Note: This requires MongoDB Atlas Vector Search index
        # For local MongoDB, use compute_similarity in application layer
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": f"{element_type}_vector_index",
                        "path": embedding_field,
                        "queryVector": query_embedding,
                        "numCandidates": top_k * 10,
                        "limit": top_k
                    }
                }
            ]
            return list(collection.aggregate(pipeline))
        except Exception as e:
            print_warning(f"Vector search not available: {e}")
            print_info("Falling back to brute-force similarity search...")
            return self._brute_force_vector_search(
                query_embedding, element_type, top_k, embedding_field
            )
    
    def _brute_force_vector_search(self, query_embedding: List[float],
                                   element_type: str, top_k: int,
                                   embedding_field: str) -> List[Dict]:
        """Brute-force vector similarity (for local MongoDB)"""
        import numpy as np
        
        collection = self.figures if element_type == 'figure' else self.tables
        
        # Get all documents with embeddings
        docs = list(collection.find({embedding_field: {"$exists": True}}))
        
        # Compute similarities
        query_vec = np.array(query_embedding)
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        similarities = []
        for doc in docs:
            # Navigate nested field
            emb = doc
            for field in embedding_field.split('.'):
                emb = emb.get(field, {})
            
            if isinstance(emb, list) and len(emb) > 0:
                doc_vec = np.array(emb)
                doc_vec = doc_vec / np.linalg.norm(doc_vec)
                sim = float(np.dot(query_vec, doc_vec))
                similarities.append((sim, doc))
        
        # Sort and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [{"similarity": sim, **doc} for sim, doc in similarities[:top_k]]
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            "papers": self.papers.count_documents({}),
            "figures": self.figures.count_documents({}),
            "tables": self.tables.count_documents({}),
            "figures_with_embeddings": self.figures.count_documents(
                {"embeddings": {"$exists": True}}
            ),
            "tables_with_embeddings": self.tables.count_documents(
                {"embeddings": {"$exists": True}}
            ),
            "figures_with_analysis": self.figures.count_documents(
                {"analysis": {"$exists": True}}
            ),
            "tables_with_analysis": self.tables.count_documents(
                {"analysis": {"$exists": True}}
            )
        }
    
    def print_statistics(self):
        """Print database statistics"""
        stats = self.get_statistics()
        print("\n" + "=" * 50)
        print("  DATABASE STATISTICS")
        print("=" * 50)
        print(f"Papers:                    {stats['papers']}")
        print(f"Figures:                   {stats['figures']}")
        print(f"Tables:                    {stats['tables']}")
        print(f"Figures with embeddings:   {stats['figures_with_embeddings']}")
        print(f"Tables with embeddings:    {stats['tables_with_embeddings']}")
        print(f"Figures with analysis:     {stats['figures_with_analysis']}")
        print(f"Tables with analysis:      {stats['tables_with_analysis']}")
        print("=" * 50 + "\n")
    
    # ==================== CLEANUP ====================
    
    def clear_collection(self, collection_name: str):
        """Clear a collection (use with caution!)"""
        if collection_name not in ["papers", "figures", "tables"]:
            print_error(f"Invalid collection name: {collection_name}")
            return
        
        self.db[collection_name].delete_many({})
        print_warning(f"Cleared collection: {collection_name}")
    
    def clear_database(self):
        """Clear entire database (use with caution!)"""
        self.papers.delete_many({})
        self.figures.delete_many({})
        self.tables.delete_many({})
        print_warning("Cleared entire database!")
    
    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        print_info("MongoDB connection closed")


# Convenience function
def get_mongodb_handler(uri: Optional[str] = None, db_name: str = "multimodal_rnd") -> MongoDBHandler:
    """Get MongoDB handler instance"""
    return MongoDBHandler(uri=uri, db_name=db_name)


if __name__ == "__main__":
    # Quick test
    try:
        handler = MongoDBHandler()
        handler.print_statistics()
        handler.close()
    except Exception as e:
        print_error(f"MongoDB test failed: {e}")
        print_info("Make sure MongoDB is running: mongod --dbpath /path/to/data")