import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Setup Logging
logger = logging.getLogger("VectorDB")

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    logger.error("chromadb is not installed. Please run 'pip install chromadb'")

class VectorDBService:
    """
    Centralized service for managing vector embeddings using ChromaDB.
    
    Provides persistent storage and efficient semantic search for multiple collections:
    - continental_dishes (512-dim)
    - indian_dishes (512-dim)
    - food_groups (512-dim)
    """
    
    DB_PATH = Path(__file__).parent.parent.parent / "data" / "chroma"
    
    def __init__(self):
        """Initialize ChromaDB client with persistent storage."""
        if not self.DB_PATH.exists():
            self.DB_PATH.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Initializing ChromaDB at {self.DB_PATH}")
        
        try:
            self.client = chromadb.PersistentClient(path=str(self.DB_PATH))
            logger.info("ChromaDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

    def get_or_create_collection(self, name: str):
        """Get an existing collection or create a new one."""
        if self.client is None:
            return None
        return self.client.get_or_create_collection(name=name)

    def upsert_vectors(self, collection_name: str, ids: List[str], embeddings: List[List[float]], metadatas: Optional[List[Dict]] = None):
        """Upsert vectors into a specific collection."""
        collection = self.get_or_create_collection(collection_name)
        if collection:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"Upserted {len(ids)} vectors into '{collection_name}'.")

    def query(self, collection_name: str, query_embeddings: List[List[float]], n_results: int = 5):
        """Search for the nearest neighbors in a collection."""
        collection = self.get_or_create_collection(collection_name)
        if collection:
            return collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
        return None

# GLOBAL SINGLETON INSTANCE
_vector_db = None

def get_vector_db_service():
    global _vector_db
    if _vector_db is None:
        _vector_db = VectorDBService()
    return _vector_db

if __name__ == "__main__":
    db = get_vector_db_service()
    if db.client:
        print("VectorDB service is ready.")
    else:
        print("VectorDB service failed to initialize.")
