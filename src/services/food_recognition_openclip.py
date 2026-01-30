"""
OpenCLIP-based Food Classification Service

This module implements a high-performance food classifier using OpenCLIP.
Optimized for speed with precomputed text embeddings and hierarchical classification.

Architecture:
- Model: ViT-B-32 with laion2b_s34b_b79k pretrained weights
- Labels: Dynamically loaded from TXT files
- Classification: Hierarchical (Cuisine → Food Group → Optional Dish)

Speed Optimizations:
1. Precompute text embeddings ONCE at startup
2. Use torch.no_grad() everywhere during inference
3. GPU acceleration when available
4. Minimal batch sizes (1-4 crops)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import torch
from PIL import Image
from services.vector_db import get_vector_db_service

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ClassificationResult:
    """Result from a single classification."""
    label: str
    confidence: float
    category: str  # "cuisine", "food_group", or "dish"


@dataclass
class HierarchicalResult:
    """Result from hierarchical classification pipeline."""
    cuisine: str
    cuisine_confidence: float
    food_group: str
    food_group_confidence: float
    # Dish name from expanded_indian_food_3000_plus.txt
    dish: Optional[str] = None
    dish_confidence: Optional[float] = None
    # Top-K predictions for food group and dish
    top_k_food_groups: Optional[List[Dict]] = None
    top_k_dishes: Optional[List[Dict]] = None


# =============================================================================
# LABEL PARSER
# =============================================================================

class LabelFileParser:
    """
    Parses TXT files containing classification labels.
    
    Format expected:
    # Category Name - description
    prompt 1 for category
    prompt 2 for category
    ...
    
    # Next Category
    prompt 1
    ...
    """
    
    @staticmethod
    def parse_file(filepath: str) -> Dict[str, List[str]]:
        """
        Parse a label TXT file into category -> prompts mapping.
        
        Args:
            filepath: Path to TXT file
            
        Returns:
            Dict mapping category names to list of prompts
        """
        categories = {}
        current_category = None
        current_prompts = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Check for category header (starts with #)
                    if line.startswith('#'):
                        # Save previous category if exists
                        if current_category and current_prompts:
                            categories[current_category] = current_prompts
                        
                        # Extract category name (text after # before any dash/description)
                        category_text = line[1:].strip()
                        if ' - ' in category_text:
                            category_text = category_text.split(' - ')[0].strip()
                        elif '-' in category_text:
                            category_text = category_text.split('-')[0].strip()
                        
                        # Only set if it's a valid category (not a file header)
                        if category_text and not category_text.lower().startswith(('food', 'cuisine', '=')):
                            current_category = category_text
                            current_prompts = []
                    else:
                        # It's a prompt line
                        if current_category:
                            current_prompts.append(line)
                
                # Save last category
                if current_category and current_prompts:
                    categories[current_category] = current_prompts
                    
        except FileNotFoundError:
            logger.error(f"Label file not found: {filepath}")
        except Exception as e:
            logger.error(f"Error parsing label file {filepath}: {e}")
        
        return categories
    
    @staticmethod
    def get_file_mtime(filepath: str) -> float:
        """Get file modification time for change detection."""
        try:
            return os.path.getmtime(filepath)
        except OSError:
            return 0.0


# =============================================================================
# OPENCLIP CLASSIFIER
# =============================================================================

class OpenCLIPClassifier:
    """
    OpenCLIP-based food classifier with precomputed embeddings.
    
    Key Features:
    - Uses ViT-B-32 with laion2b_s34b_b79k (MANDATORY - DO NOT CHANGE)
    - Precomputes text embeddings at startup for 3-4x speedup
    - Hierarchical classification: Cuisine → Food Group
    - Dynamic TXT file loading with hot-reload support
    - GPU acceleration when available
    
    Usage:
        classifier = OpenCLIPClassifier()
        result = classifier.classify_hierarchical(pil_image)
        print(f"Cuisine: {result.cuisine}, Food Group: {result.food_group}")
    """
    
    # Default paths relative to project root
    DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data"
    CUISINE_FILE = "CLIP_Cuisine.txt"
    FOOD_GROUPS_FILE = "CLIP_Food_Groups.txt"
    DISH_FILE = "expanded_indian_food_3000_plus.txt"
    
    def __init__(
        self,
        data_dir: Optional[str] = None,
        device: Optional[str] = None,
        enable_dish_classification: bool = True  # Now defaults to True
    ):
        """
        Initialize OpenCLIP classifier.
        
        Args:
            data_dir: Directory containing TXT label files
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            enable_dish_classification: If True, load dish names from TXT file
        """
        # SPEED: Set device (GPU if available)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"OpenCLIP using device: {self.device}")
        
        # Data directory
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        
        # Model components (lazy loaded)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._model_loaded = False
        
        # Label storage
        self.cuisine_labels: Dict[str, List[str]] = {}
        self.food_group_labels: Dict[str, List[str]] = {}
        # Dish labels loaded from expanded_indian_food_3000_plus.txt
        self.dish_labels: List[str] = []
        self.enable_dish_classification = enable_dish_classification
        
        # SPEED: Precomputed text embeddings (cached tensors)
        self.cuisine_embeddings: Optional[torch.Tensor] = None
        self.cuisine_names: List[str] = []
        self.food_group_embeddings: Optional[torch.Tensor] = None
        self.food_group_names: List[str] = []
        
        # VectorDB integration
        self.vdb = get_vector_db_service()
        self.dish_collection = "indian_dishes"
        
        # File modification times for hot-reload detection
        self._file_mtimes: Dict[str, float] = {}
        
        # Load everything at init
        self._load_model()
        self._load_labels()
        self._precompute_embeddings()
    
    def _load_model(self) -> None:
        """
        Load OpenCLIP model with MANDATORY configuration.
        
        IMPORTANT: These exact parameters are required - DO NOT CHANGE.
        - Model: ViT-B-32
        - Pretrained: laion2b_s34b_b79k
        """
        try:
            import open_clip
            
            logger.info("Loading OpenCLIP ViT-B-32 with laion2b_s34b_b79k...")
            
            # MANDATORY MODEL CONFIGURATION - DO NOT CHANGE
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', 
                pretrained='laion2b_s34b_b79k'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
            
            # Move model to device
            self.model = self.model.to(self.device)
            
            # Set to evaluation mode (SPEED: disables dropout, etc.)
            self.model.eval()
            
            self._model_loaded = True
            logger.info("OpenCLIP model loaded successfully")
            
        except ImportError:
            logger.error("open_clip not installed. Run: pip install open_clip_torch")
            raise
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            raise
    
    def _load_labels(self) -> None:
        """Load all label files (cuisine, food groups, and optionally dishes)."""
        
        # Load cuisine labels
        cuisine_path = self.data_dir / self.CUISINE_FILE
        if cuisine_path.exists():
            self.cuisine_labels = LabelFileParser.parse_file(str(cuisine_path))
            self._file_mtimes[str(cuisine_path)] = LabelFileParser.get_file_mtime(str(cuisine_path))
            logger.info(f"Loaded {len(self.cuisine_labels)} cuisine categories")
        else:
            logger.warning(f"Cuisine file not found: {cuisine_path}")
            # Fallback labels
            self.cuisine_labels = {
                "Indian Food": ["a photo of Indian food", "Indian cuisine"],
                "Continental Food": ["a photo of continental food", "Western cuisine"]
            }
        
        # Load food group labels
        food_groups_path = self.data_dir / self.FOOD_GROUPS_FILE
        if food_groups_path.exists():
            self.food_group_labels = LabelFileParser.parse_file(str(food_groups_path))
            self._file_mtimes[str(food_groups_path)] = LabelFileParser.get_file_mtime(str(food_groups_path))
            logger.info(f"Loaded {len(self.food_group_labels)} food group categories")
        else:
            logger.warning(f"Food groups file not found: {food_groups_path}")
        
        # Load dish labels from expanded_indian_food_3000_plus.txt
        # These are ALWAYS loaded to provide the actual food dish names
        dish_path = self.data_dir / self.DISH_FILE
        if dish_path.exists():
            self.dish_labels = self._load_dish_file(str(dish_path))
            self._file_mtimes[str(dish_path)] = LabelFileParser.get_file_mtime(str(dish_path))
            logger.info(f"Loaded {len(self.dish_labels)} dish labels from {self.DISH_FILE}")
        else:
            logger.warning(f"Dish file not found: {dish_path}")
    
    def _load_dish_file(self, filepath: str) -> List[str]:
        """Load simple line-based dish list."""
        dishes = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dishes.append(line)
        except Exception as e:
            logger.error(f"Error loading dish file: {e}")
        return dishes
    
    def _precompute_embeddings(self) -> None:
        """
        Precompute and cache text embeddings for all labels.
        
        SPEED CRITICAL: This is called ONCE at startup.
        Text embeddings are expensive to compute, so we cache them.
        """
        logger.info("Precomputing text embeddings (this happens once at startup)...")
        
        # SPEED: No gradient computation needed for inference
        with torch.no_grad():
            # 1. Cuisine embeddings (average multiple prompts per cuisine)
            self.cuisine_names = []
            cuisine_embeds = []
            
            for cuisine_name, prompts in self.cuisine_labels.items():
                self.cuisine_names.append(cuisine_name)
                
                # Tokenize all prompts for this cuisine
                tokens = self.tokenizer(prompts).to(self.device)
                
                # Encode and average (ensemble)
                embeds = self.model.encode_text(tokens)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)  # Normalize
                avg_embed = embeds.mean(dim=0)
                avg_embed = avg_embed / avg_embed.norm()  # Re-normalize after averaging
                
                cuisine_embeds.append(avg_embed)
            
            if cuisine_embeds:
                self.cuisine_embeddings = torch.stack(cuisine_embeds)
                logger.info(f"Precomputed {len(self.cuisine_names)} cuisine embeddings")
            
            # 2. Food group embeddings (average multiple prompts per group)
            self.food_group_names = []
            group_embeds = []
            
            for group_name, prompts in self.food_group_labels.items():
                self.food_group_names.append(group_name)
                
                tokens = self.tokenizer(prompts).to(self.device)
                embeds = self.model.encode_text(tokens)
                embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                avg_embed = embeds.mean(dim=0)
                avg_embed = avg_embed / avg_embed.norm()
                
                group_embeds.append(avg_embed)
            
            if group_embeds:
                self.food_group_embeddings = torch.stack(group_embeds)
                logger.info(f"Precomputed {len(self.food_group_names)} food group embeddings")
            
            # 3. Dish embeddings - Pushed to VectorDB
            if self.dish_labels:
                collection = self.vdb.get_or_create_collection(self.dish_collection)
                if collection and collection.count() >= len(self.dish_labels):
                    logger.info(f"✓ VectorDB collection '{self.dish_collection}' already populated.")
                else:
                    logger.info(f"Precomputing {len(self.dish_labels)} dish embeddings for VectorDB...")
                    batch_size = 100
                    all_dish_embeds = []
                    all_ids = []
                    
                    for i in range(0, len(self.dish_labels), batch_size):
                        batch = self.dish_labels[i:i + batch_size]
                        prompts = [f"a photo of {dish}" for dish in batch]
                        tokens = self.tokenizer(prompts).to(self.device)
                        embeds = self.model.encode_text(tokens)
                        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
                        
                        all_dish_embeds.append(embeds.cpu())
                        all_ids.extend([f"dish_{j}" for j in range(i, i + len(batch))])
                    
                    if all_dish_embeds:
                        embeddings_list = torch.cat(all_dish_embeds, dim=0).tolist()
                        metadatas = [{"name": name} for name in self.dish_labels]
                        self.vdb.upsert_vectors(
                            collection_name=self.dish_collection,
                            ids=all_ids,
                            embeddings=embeddings_list,
                            metadatas=metadatas
                        )
                        logger.info(f"✓ Pushed {len(self.dish_labels)} dish embeddings to VectorDB")
            else:
                logger.warning("No dish labels loaded - dish names will not be available")
        
        logger.info("Text embedding precomputation complete")
    
    def check_and_reload_if_needed(self) -> bool:
        """
        Check if TXT files have changed and reload embeddings if needed.
        
        Returns:
            True if embeddings were reloaded, False otherwise
        """
        needs_reload = False
        
        for filepath, old_mtime in self._file_mtimes.items():
            current_mtime = LabelFileParser.get_file_mtime(filepath)
            if current_mtime > old_mtime:
                logger.info(f"Detected change in {filepath}, reloading...")
                needs_reload = True
                break
        
        if needs_reload:
            self._load_labels()
            self._precompute_embeddings()
            return True
        
        return False
    
    def reload_embeddings(self) -> None:
        """Force reload of labels and embeddings."""
        logger.info("Force reloading labels and embeddings...")
        self._load_labels()
        self._precompute_embeddings()
    
    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image to embedding.
        
        SPEED: Uses torch.no_grad() to skip gradient computation.
        """
        # Preprocess image
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # SPEED: No gradient needed for inference
        with torch.no_grad():
            image_embedding = self.model.encode_image(image_tensor)
            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        
        return image_embedding
    
    def classify_cuisine(
        self, 
        image: Image.Image, 
        top_k: int = 2
    ) -> List[ClassificationResult]:
        """
        Stage 1: Classify cuisine (Indian vs Continental).
        
        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of ClassificationResult sorted by confidence
        """
        if self.cuisine_embeddings is None or len(self.cuisine_names) == 0:
            logger.warning("No cuisine embeddings available")
            return []
        
        # Encode image
        image_embedding = self._encode_image(image)
        
        # SPEED: Use cached text embeddings, compute similarity
        with torch.no_grad():
            similarities = (image_embedding @ self.cuisine_embeddings.T).squeeze(0)
            
            # Convert to probabilities
            probs = torch.softmax(similarities * 100, dim=0)  # Scale for better distribution
            
            # Get top-k
            top_probs, top_indices = probs.topk(min(top_k, len(self.cuisine_names)))
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results.append(ClassificationResult(
                label=self.cuisine_names[idx],
                confidence=float(prob),
                category="cuisine"
            ))
        
        return results
    
    def classify_food_group(
        self, 
        image: Image.Image, 
        top_k: int = 5
    ) -> List[ClassificationResult]:
        """
        Stage 2: Classify food group.
        
        Args:
            image: PIL Image to classify
            top_k: Number of top predictions to return
            
        Returns:
            List of ClassificationResult sorted by confidence
        """
        if self.food_group_embeddings is None or len(self.food_group_names) == 0:
            logger.warning("No food group embeddings available")
            return []
        
        # Encode image
        image_embedding = self._encode_image(image)
        
        # SPEED: Use cached text embeddings
        with torch.no_grad():
            similarities = (image_embedding @ self.food_group_embeddings.T).squeeze(0)
            probs = torch.softmax(similarities * 100, dim=0)
            top_probs, top_indices = probs.topk(min(top_k, len(self.food_group_names)))
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            results.append(ClassificationResult(
                label=self.food_group_names[idx],
                confidence=float(prob),
                category="food_group"
            ))
        
        return results
    
    def classify_dish(
        self, 
        image: Image.Image, 
        top_k: int = 5
    ) -> List[ClassificationResult]:
        """
        Stage 3: Classify specific dish using VectorDB.
        """
        if not self.dish_labels:
            logger.warning("Dish labels not loaded")
            return []
        
        # Encode image
        image_embedding = self._encode_image(image)
        
        # Search VectorDB
        query_embeddings = image_embedding.cpu().tolist()
        results = self.vdb.query(
            collection_name=self.dish_collection,
            query_embeddings=query_embeddings,
            n_results=top_k
        )
        
        if not results or not results['ids'][0]:
            return []
            
        results_list = []
        for i in range(len(results['ids'][0])):
            results_list.append(ClassificationResult(
                label=results['metadatas'][0][i]['name'],
                confidence=float(1.0 - results['distances'][0][i]), # Approximate
                category="dish"
            ))
        
        return results_list
    
    def classify_hierarchical(
        self, 
        image: Image.Image,
        top_k: int = 3,
        include_dish: bool = True  # Now defaults to True to always get dish names
    ) -> HierarchicalResult:
        """
        Full hierarchical classification: Cuisine → Food Group → Dish.
        
        This is the main entry point for food classification.
        The dish name comes from expanded_indian_food_3000_plus.txt
        
        Args:
            image: PIL Image to classify
            top_k: Number of predictions to keep at each level
            include_dish: If True, predict specific dish name (default: True)
            
        Returns:
            HierarchicalResult with cuisine, food group, and dish name
        """
        # Stage 1: Cuisine
        cuisine_results = self.classify_cuisine(image, top_k=2)
        if not cuisine_results:
            return HierarchicalResult(
                cuisine="Unknown",
                cuisine_confidence=0.0,
                food_group="Unknown",
                food_group_confidence=0.0
            )
        
        top_cuisine = cuisine_results[0]
        
        # Stage 2: Food Group
        food_group_results = self.classify_food_group(image, top_k=top_k)
        if not food_group_results:
            return HierarchicalResult(
                cuisine=top_cuisine.label,
                cuisine_confidence=top_cuisine.confidence,
                food_group="Unknown",
                food_group_confidence=0.0
            )
        
        top_food_group = food_group_results[0]
        
        # Stage 3: Dish name from expanded_indian_food_3000_plus.txt
        # This is now ALWAYS computed to provide actual food names
        dish_results = []
        if include_dish and self.dish_embeddings is not None:
            dish_results = self.classify_dish(image, top_k=top_k)
        
        # Build result with all classifications
        result = HierarchicalResult(
            cuisine=top_cuisine.label,
            cuisine_confidence=top_cuisine.confidence,
            food_group=top_food_group.label,
            food_group_confidence=top_food_group.confidence,
            # Dish name from expanded food list
            dish=dish_results[0].label if dish_results else None,
            dish_confidence=dish_results[0].confidence if dish_results else None,
            # Top-K predictions for both levels
            top_k_food_groups=[
                {"label": r.label, "confidence": r.confidence} 
                for r in food_group_results
            ],
            top_k_dishes=[
                {"label": r.label, "confidence": r.confidence} 
                for r in dish_results
            ] if dish_results else None
        )
        
        return result
    
    def classify_crop(
        self, 
        image: Image.Image, 
        bbox: Tuple[int, int, int, int],
        top_k: int = 3
    ) -> HierarchicalResult:
        """
        Classify a cropped region of an image.
        
        Args:
            image: Full PIL Image
            bbox: Bounding box (x1, y1, x2, y2)
            top_k: Number of predictions to return
            
        Returns:
            HierarchicalResult for the cropped region
        """
        # Crop the region
        x1, y1, x2, y2 = bbox
        cropped = image.crop((x1, y1, x2, y2))
        
        # Classify the crop
        return self.classify_hierarchical(cropped, top_k=top_k)
    
    def is_available(self) -> bool:
        """Check if classifier is ready."""
        return self._model_loaded and self.cuisine_embeddings is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "model_loaded": self._model_loaded,
            "device": self.device,
            "num_cuisines": len(self.cuisine_names),
            "num_food_groups": len(self.food_group_names),
            "num_dishes": len(self.dish_labels) if self.enable_dish_classification else 0,
            "cuisine_labels": self.cuisine_names,
            "food_group_labels": self.food_group_names,
        }


# =============================================================================
# SINGLETON PATTERN
# =============================================================================

_classifier_instance: Optional[OpenCLIPClassifier] = None


def get_openclip_classifier(
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    enable_dish_classification: bool = False
) -> OpenCLIPClassifier:
    """
    Get or create the global OpenCLIP classifier instance.
    
    Args:
        data_dir: Directory containing TXT label files
        device: Device to use ('cuda', 'cpu', or None for auto)
        enable_dish_classification: Enable dish-level classification
        
    Returns:
        OpenCLIPClassifier singleton instance
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        _classifier_instance = OpenCLIPClassifier(
            data_dir=data_dir,
            device=device,
            enable_dish_classification=enable_dish_classification
        )
    
    return _classifier_instance


def classify_food_image(
    image: Image.Image,
    top_k: int = 3.
) -> HierarchicalResult:
    """
    Convenience function to classify a food image.
    
    Args:
        image: PIL Image to classify
        top_k: Number of predictions per level
        
    Returns:
        HierarchicalResult with cuisine and food group
    """
    classifier = get_openclip_classifier()
    return classifier.classify_hierarchical(image, top_k=top_k)
