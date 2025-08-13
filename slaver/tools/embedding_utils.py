#!/usr/bin/env python
# coding=utf-8
"""
Enhanced Embedding and Vector Similarity Utilities for Tool Selection

This module provides intelligent tool selection based on semantic understanding,
automatic categorization, and advanced matching algorithms. It uses TF-IDF
vectorization and cosine similarity to find the most relevant tools for given tasks.

Features:
- Semantic tool categorization (navigation, manipulation, sensing, etc.)
- Keyword extraction and synonym expansion
- Configurable similarity thresholds
- Enhanced tool descriptions with metadata
- Category-based filtering for better relevance
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import json
import re
import os
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Default configuration values as fallback
DEFAULT_CONFIG = {
    "embedding": {
        "method": "enhanced_tfidf",
        "top_k": 5,
        "similarity_threshold": 0.05,
        "debug": True,
        "use_category_filtering": True,
        "min_tools": 2,
        "max_tools": 3,
        "name_weight": 0.4,
        "description_weight": 0.6,
        "fuzzy_match": True,
        "synonym_match": True
    },
    "tool_selection": {
        "default_tool_count": 5,
        "enable_caching": True,
        "cache_expiration": 3600,
        "log_selection": True
    },
    "category_keywords": {
        "navigation": ["navigate", "move", "go", "travel", "position", "location", "target", "destination"],
        "manipulation": ["grasp", "pick", "place", "hold", "carry", "move", "rotate", "apply", "force"],
        "sensing": ["scan", "detect", "observe", "photo", "image", "camera", "vision", "check", "status"],
        "cleaning": ["clean", "wipe", "wash", "remove", "dust", "surface", "maintenance"],
        "organization": ["organize", "arrange", "sort", "categorize", "group", "order", "structure"],
        "pose": ["pose", "position", "stance", "posture", "ready", "standby", "set"]
    },
    "synonym_mappings": {
        "pick": ["grasp", "take", "hold", "seize"],
        "place": ["put", "set", "position", "drop"],
        "move": ["navigate", "go", "travel", "walk"],
        "scan": ["observe", "look", "examine", "inspect"],
        "clean": ["wipe", "wash", "clear", "remove"],
        "organize": ["arrange", "sort", "order", "categorize"]
    },
    "stop_words": [
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
        "has", "had", "do", "does", "did", "will", "would", "could", "should",
        "may", "might", "can", "this", "that", "these", "those", "i", "you",
        "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"
    ]
}

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the configuration file. If None, uses hardcoded path.
        
    Returns:
        Dictionary containing the configuration
    """
    # Always use hardcoded path relative to this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "config.yaml")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        else:
            logger.warning(f"Configuration file not found at {config_path}, using defaults")
            return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return DEFAULT_CONFIG

# Load configuration once at module import
_config = load_config()

# Export configuration as module-level variables
EMBEDDING_CONFIG = _config.get("embedding", DEFAULT_CONFIG["embedding"])
TOOL_SELECTION_CONFIG = _config.get("tool_selection", DEFAULT_CONFIG["tool_selection"])
CATEGORY_KEYWORDS = _config.get("category_keywords", DEFAULT_CONFIG["category_keywords"])
SYNONYM_MAPPINGS = _config.get("synonym_mappings", DEFAULT_CONFIG["synonym_mappings"])
STOP_WORDS = _config.get("stop_words", DEFAULT_CONFIG["stop_words"])


class EnhancedEmbeddingManager:
    """Enhanced embedding manager with semantic understanding and categorization"""
    
    def __init__(self, method: str = None):
        """
        Initialize the enhanced embedding manager
        
        Args:
            method: Embedding method to use (defaults to config value)
        """
        # Use config values with fallbacks
        self.method = method or EMBEDDING_CONFIG.get("method", "enhanced_tfidf")
        self.vectorizer = None
        self.tool_embeddings = {}
        self.tool_descriptions = {}
        self.tool_categories = {}
        self.tool_keywords = {}
        self.tool_schemas = {}
        
        # Load configuration from config file
        self.category_keywords = CATEGORY_KEYWORDS
        self.synonyms = SYNONYM_MAPPINGS
        
        # Load embedding configuration
        self.top_k = EMBEDDING_CONFIG.get("top_k", 5)
        self.similarity_threshold = EMBEDDING_CONFIG.get("similarity_threshold", 0.05)
        self.debug = EMBEDDING_CONFIG.get("debug", True)
        self.use_category_filtering = EMBEDDING_CONFIG.get("use_category_filtering", True)
        self.min_tools = EMBEDDING_CONFIG.get("min_tools", 2)
        self.max_tools = EMBEDDING_CONFIG.get("max_tools", 8)
        self.name_weight = EMBEDDING_CONFIG.get("name_weight", 0.4)
        self.description_weight = EMBEDDING_CONFIG.get("description_weight", 0.6)
        self.fuzzy_match = EMBEDDING_CONFIG.get("fuzzy_match", True)
        self.synonym_match = EMBEDDING_CONFIG.get("synonym_match", True)
        
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        words = text.split()
        
        # Filter stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in STOP_WORDS]
        
        return keywords
    
    def _categorize_tool(self, tool_name: str, description: str) -> str:
        """Automatically categorize a tool based on name and description"""
        text = f"{tool_name} {description}".lower()
        
        # Calculate matching score for each category
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            category_scores[category] = score
        
        # Return the category with highest score
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0] if best_category[1] > 0 else "general"
        
        return "general"
    
    def _enhance_description(self, tool_name: str, description: str, schema: Dict = None) -> str:
        """Create an enhanced description with semantic information"""
        # Base description
        enhanced_parts = [description]
        
        # Add semantic information from tool name
        name_keywords = self._extract_keywords(tool_name.replace("_", " "))
        enhanced_parts.append(f"Tool name indicates: {' '.join(name_keywords)}")
        
        # Add category information
        category = self._categorize_tool(tool_name, description)
        enhanced_parts.append(f"Category: {category}")
        
        # Add keywords
        keywords = self._extract_keywords(description)
        enhanced_parts.append(f"Keywords: {' '.join(keywords)}")
        
        # Add synonym expansion
        expanded_keywords = []
        for keyword in keywords:
            if keyword in self.synonyms:
                expanded_keywords.extend(self.synonyms[keyword])
        
        if expanded_keywords:
            enhanced_parts.append(f"Related terms: {' '.join(set(expanded_keywords))}")
        
        # Add parameter information
        if schema:
            param_info = self._extract_parameter_semantics(schema)
            if param_info:
                enhanced_parts.append(f"Parameters: {param_info}")
        
        return " | ".join(enhanced_parts)
    
    def _extract_parameter_semantics(self, schema: Dict) -> str:
        """Extract semantic information from parameter schema"""
        if not schema or "properties" not in schema:
            return ""
        
        param_descriptions = []
        for param_name, param_def in schema.get("properties", {}).items():
            param_type = param_def.get("type", "unknown")
            param_desc = param_def.get("description", "")
            
            if param_desc:
                param_descriptions.append(f"{param_name}({param_type}): {param_desc}")
            else:
                param_descriptions.append(f"{param_name}({param_type})")
        
        return "; ".join(param_descriptions)
    
    def _create_enhanced_embedding(self, text: str) -> np.ndarray:
        """Create enhanced embedding with better feature extraction"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=2000,  # Increased feature count
                stop_words='english',
                ngram_range=(1, 3),  # Support 1-3 word combinations
                min_df=1,
                max_df=1.0  # Allow all document frequencies
            )
        
        # Ensure vectorizer is trained
        if not hasattr(self.vectorizer, 'vocabulary_'):
            # If not trained, train first
            all_texts = list(self.tool_descriptions.values())
            if all_texts:
                self.vectorizer.fit(all_texts)
        
        # Transform text
        embedding = self.vectorizer.transform([text]).toarray()
        return embedding.flatten()
    
    def add_tool(self, tool_name: str, tool_description: str, tool_schema: Dict = None):
        """Add a tool with enhanced semantic information"""
        # Create enhanced description
        enhanced_description = self._enhance_description(tool_name, tool_description, tool_schema)
        
        # Store various information
        self.tool_descriptions[tool_name] = enhanced_description
        self.tool_schemas[tool_name] = tool_schema
        
        # Determine category
        category = self._categorize_tool(tool_name, tool_description)
        self.tool_categories[tool_name] = category
        
        # Extract keywords
        keywords = self._extract_keywords(tool_description)
        self.tool_keywords[tool_name] = keywords
        
        # Generate embedding vector
        if self.method == "enhanced_tfidf":
            # Reset vectorizer to retrain
            self.vectorizer = None
            embedding = self._create_enhanced_embedding(enhanced_description)
        else:
            embedding = self._create_enhanced_embedding(enhanced_description)
        
        self.tool_embeddings[tool_name] = embedding
        
        logger.debug(f"Added tool '{tool_name}' (Category: {category}) with enhanced embedding")
    
    def get_task_embedding(self, task: str) -> np.ndarray:
        """Get enhanced embedding for a task"""
        # Enhance task as well
        task_keywords = self._extract_keywords(task)
        enhanced_task = f"{task} | Keywords: {' '.join(task_keywords)}"
        return self._create_enhanced_embedding(enhanced_task)
    
    def find_most_similar_tools(self, task: str, top_k: int = 3, 
                               category_filter: str = None) -> List[Tuple[str, float]]:
        """Find most similar tools with optional category filtering"""
        if not self.tool_embeddings:
            logger.warning("No tools available for similarity search")
            return []
        
        # Get task embedding
        task_embedding = self.get_task_embedding(task)
        
        # Calculate similarities
        similarities = []
        for tool_name, tool_embedding in self.tool_embeddings.items():
            # Category filtering
            if category_filter and self.tool_categories.get(tool_name) != category_filter:
                continue
            
            # Ensure embedding vector dimensions match
            if task_embedding.shape != tool_embedding.shape:
                min_size = min(task_embedding.shape[0], tool_embedding.shape[0])
                task_emb = task_embedding[:min_size]
                tool_emb = tool_embedding[:min_size]
            else:
                task_emb = task_embedding
                tool_emb = tool_embedding
            
            # Calculate cosine similarity
            similarity = cosine_similarity([task_emb], [tool_emb])[0][0]
            similarities.append((tool_name, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_tools = similarities[:top_k]
        
        logger.info(f"Top {top_k} similar tools for task '{task[:50]}...':")
        for tool_name, score in top_tools:
            category = self.tool_categories.get(tool_name, "unknown")
            logger.info(f"  {tool_name} ({category}): {score:.4f}")
        
        return top_tools
    
    def get_tools_by_names(self, tool_names: List[str]) -> List[Dict]:
        """Get tool definitions by names"""
        tools = []
        for name in tool_names:
            # First try to get the original tool definition if available
            if hasattr(self, '_original_tools') and name in self._original_tools:
                tools.append(self._original_tools[name])
            elif name in self.tool_descriptions:
                # Fallback: rebuild tool definition
                tool_def = {
                    "function": {
                        "name": name,
                        "description": self.tool_descriptions[name].split(" | ")[0]  # Only original description
                    }
                }
                
                # Add schema information
                if name in self.tool_schemas and self.tool_schemas[name]:
                    tool_def["input_schema"] = self.tool_schemas[name]
                
                tools.append(tool_def)
        
        return tools
    
    def get_tools_by_category(self, category: str) -> List[str]:
        """Get all tools in a specific category"""
        return [name for name, cat in self.tool_categories.items() if cat == category]
    
    def analyze_task_requirements(self, task: str) -> Dict:
        """Analyze task to determine required tool categories"""
        task_lower = task.lower()
        required_categories = []
        
        for category, keywords in self.category_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                required_categories.append(category)
        
        return {
            "required_categories": required_categories,
            "task_keywords": self._extract_keywords(task),
            "suggested_tools": self._suggest_tools_by_categories(required_categories)
        }
    
    def _suggest_tools_by_categories(self, categories: List[str]) -> List[str]:
        """Suggest tools based on required categories"""
        suggested = []
        for category in categories:
            category_tools = self.get_tools_by_category(category)
            suggested.extend(category_tools)
        return list(set(suggested))
    
    def clear_tools(self):
        """Clear all stored tools and embeddings"""
        self.tool_embeddings.clear()
        self.tool_descriptions.clear()
        self.tool_categories.clear()
        self.tool_keywords.clear()
        self.tool_schemas.clear()
        self.vectorizer = None
        # Clear original tools if they exist
        if hasattr(self, '_original_tools'):
            self._original_tools.clear()
        logger.info("Cleared all tools and embeddings")


# Global instance
enhanced_embedding_manager = EnhancedEmbeddingManager()


def add_tools_to_enhanced_manager(tools: List[Dict]):
    """Add tools to the enhanced embedding manager"""
    # Store the original tools for later retrieval
    enhanced_embedding_manager._original_tools = {
        tool.get("function", {}).get("name", ""): tool 
        for tool in tools 
        if tool.get("function", {}).get("name", "")
    }
    
    for tool in tools:
        tool_name = tool.get("function", {}).get("name", "")
        tool_description = tool.get("function", {}).get("description", "")
        tool_schema = tool.get("input_schema")
        
        if tool_name and tool_description:
            enhanced_embedding_manager.add_tool(tool_name, tool_description, tool_schema)


def select_relevant_tools(task: str, all_tools: List[Dict], 
                         top_k: int = None, 
                         use_category_filtering: bool = True) -> List[Dict]:
    """
    Enhanced tool selection with semantic understanding and categorization
    
    Args:
        task: Task description
        all_tools: List of all available tools
        top_k: Number of top tools to select
        use_category_filtering: Whether to use category-based filtering
        
    Returns:
        List of selected tool definitions
    """
    # Load configuration from config file
    if top_k is None:
        top_k = EMBEDDING_CONFIG.get("top_k", 3)
    
    config_threshold = EMBEDDING_CONFIG.get("similarity_threshold", 0.05)
    config_debug = EMBEDDING_CONFIG.get("debug", False)
    
    if config_debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Clear and re-add tools
    enhanced_embedding_manager.clear_tools()
    add_tools_to_enhanced_manager(all_tools)
    
    # Analyze task requirements
    task_analysis = enhanced_embedding_manager.analyze_task_requirements(task)
    logger.info(f"Task analysis: {task_analysis}")
    
    # If category filtering is enabled, prioritize tools from relevant categories
    if use_category_filtering and task_analysis["required_categories"]:
        # Find most similar tools for each relevant category
        selected_tools = []
        tools_per_category = max(1, top_k // len(task_analysis["required_categories"]))
        
        for category in task_analysis["required_categories"]:
            category_tools = enhanced_embedding_manager.find_most_similar_tools(
                task, tools_per_category, category
            )
            selected_tools.extend(category_tools)
        
        # Sort by similarity and take top_k
        selected_tools.sort(key=lambda x: x[1], reverse=True)
        selected_tools = selected_tools[:top_k]
        
    else:
        # Use traditional method
        selected_tools = enhanced_embedding_manager.find_most_similar_tools(task, top_k)
    
    # Filter by similarity threshold
    filtered_tools = [(name, score) for name, score in selected_tools if score >= config_threshold]
    
    if not filtered_tools:
        logger.warning(f"No tools met similarity threshold {config_threshold}, using top {top_k}")
        filtered_tools = selected_tools[:top_k]
    
    tool_names = [tool_name for tool_name, _ in filtered_tools]
    
    # Return selected tools
    return enhanced_embedding_manager.get_tools_by_names(tool_names)

