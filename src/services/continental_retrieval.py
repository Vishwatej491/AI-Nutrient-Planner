import os
import torch
import logging
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import CLIPProcessor, CLIPModel
from services.vector_db import get_vector_db_service

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ContinentalRetrieval")

class ContinentalRetrievalSystem:
    """
    A production-grade CLIP-based retrieval system for continental dishes.
    
    Architecture:
    1. Eager loading of CLIP model.
    2. One-time precomputation of dish text embeddings using multiple prompts.
    3. Runtime image encoding and cosine similarity search.
    """
    
    MODEL_ID = "openai/clip-vit-base-patch32"
    DISHES_PATH = Path(__file__).parent.parent.parent / "data" / "dishes_continental.txt"
    CONFIDENCE_THRESHOLD = 0.20
    
    def __init__(self):
        # State variables
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.model = None
        self.processor = None
        self.dish_names: List[str] = []
        self.text_features: Optional[torch.Tensor] = None
        self.vdb = get_vector_db_service()
        self.collection_name = "continental_dishes"
        
        logger.info(f"Initializing ContinentalRetrievalSystem on {self.device} ({self.dtype})")
        
        # OFFLINE / STARTUP STAGES
        self.load_model()
        self.load_dishes_txt()
        self.build_text_index()

    def load_model(self):
        """A. Loads CLIP model and processor ONCE."""
        logger.info(f"Loading weights for {self.MODEL_ID}...")
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(self.device).to(self.dtype)
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model.eval()
        
        # Sanity check logging
        logger.info(f"Model loaded. Precision: {self.model.dtype}")

    def load_dishes_txt(self):
        """B. Reads dish names from the provided TXT file."""
        if not self.DISHES_PATH.exists():
            raise FileNotFoundError(f"Continental dishes file not found at {self.DISHES_PATH}")
            
        with open(self.DISHES_PATH, "r", encoding="utf-8") as f:
            self.dish_names = [line.strip() for line in f if line.strip()]
            
        logger.info(f"Loaded {len(self.dish_names)} continental dishes.")

    @torch.inference_mode()
    def build_text_index(self):
        """C. Generates and pushes text embeddings to ChromaDB."""
        # Check if already in VectorDB
        collection = self.vdb.get_or_create_collection(self.collection_name)
        if collection and collection.count() >= len(self.dish_names):
            logger.info(f"✓ VectorDB collection '{self.collection_name}' already populated.")
            return

        logger.info("Building text embedding index for VectorDB (Offline Stage)...")
        
        # Use only the two requested prompt templates
        templates = ["a photo of {}", "a photo of {} food"]
        
        all_features = []
        all_ids = []
        
        # Batch process for efficiency
        batch_size = 32
        for i in range(0, len(self.dish_names), batch_size):
            batch_dishes = self.dish_names[i : i + batch_size]
            
            # For each template, compute features
            template_features = []
            for template in templates:
                prompts = [template.format(dish) for dish in batch_dishes]
                inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                features = self.model.get_text_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                template_features.append(features)
            
            # Average and normalize
            averaged_batch = torch.stack(template_features).mean(dim=0)
            averaged_batch = averaged_batch / averaged_batch.norm(p=2, dim=-1, keepdim=True)
            
            # Add to list for upserting
            all_features.append(averaged_batch.cpu())
            all_ids.extend([f"cont_{j}" for j in range(i, i + len(batch_dishes))])
            
        # Push to VectorDB
        embeddings_list = torch.cat(all_features, dim=0).tolist()
        metadatas = [{"name": name} for name in self.dish_names]
        
        self.vdb.upsert_vectors(
            collection_name=self.collection_name,
            ids=all_ids,
            embeddings=embeddings_list,
            metadatas=metadatas
        )
        logger.info(f"✓ Index built and pushed to VectorDB '{self.collection_name}'")

    @torch.inference_mode()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """D. Encodes a single image into normalized CLIP space."""
        # Preprocessing (Fix resizing and normalization per model specs)
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Move inputs to correct dtype
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        
        # Forward pass
        image_features = self.model.get_image_features(**inputs)
        
        # Sanity check logging
        # logger.debug(f"Image feature shape: {image_features.shape}")
        
        # Normalization (MANDATORY)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def retrieve_top_k(self, image_features: torch.Tensor, k: int = 5) -> Dict[str, Any]:
        """E. Queries VectorDB for Nearest Neighbors."""
        # Search VectorDB
        query_embeddings = image_features.cpu().tolist()
        results = self.vdb.query(
            collection_name=self.collection_name,
            query_embeddings=query_embeddings,
            n_results=k
        )
        
        if not results or not results['ids'][0]:
            return {
                "top_k_predictions": [],
                "confidence": 0.0,
                "status": "unknown",
                "message": "No results from VectorDB"
            }
            
        predictions = []
        # distance in chroma is usually L2 or Cosine. We assume cosine for CLIP.
        # similarities = 1 - distance (if using cosine distance)
        # Note: CLIP features were normalized, so dot product = cosine similarity.
        
        for i in range(len(results['ids'][0])):
            predictions.append({
                "dish": results['metadatas'][0][i]['name'],
                "score": round(1.0 - results['distances'][0][i], 4) # Approximate cosine similarity
            })
            
        max_score = predictions[0]["score"]
        status = "ok" if max_score >= self.CONFIDENCE_THRESHOLD else "unknown"
        
        return {
            "top_k_predictions": predictions,
            "confidence": max_score,
            "status": status,
            "message": None if status == "ok" else "Unknown continental dish"
        }

    def main_inference(self, image: Any, k: int = 5) -> Dict[str, Any]:
        """F. Orchestrates the full online inference flow."""
        try:
            # Handle path vs PIL
            if isinstance(image, (str, Path)):
                image = Image.open(image).convert("RGB")
            
            # Step 1: Online Image Encoding
            img_embed = self.encode_image(image)
            
            # Step 2: Cosine Similarity Matching
            result = self.retrieve_top_k(img_embed, k=k)
            return result
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "top_k_predictions": [],
                "confidence": 0.0,
                "status": "error",
                "message": str(e)
            }

# GLOBAL SINGLETON INSTANCE
_system = None

def get_continental_retrieval_system():
    global _system
    if _system is None:
        _system = ContinentalRetrievalSystem()
    return _system

if __name__ == "__main__":
    # Quick sanity check
    sys = get_continental_retrieval_system()
    print("System initialized successfully.")
