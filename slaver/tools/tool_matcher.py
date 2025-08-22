#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tool Matcher with Semantic Embeddings

This module provides intelligent tool matching using semantic embeddings
to match tasks with the most relevant tools based on their names and descriptions.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class ToolMatcher:
    """
    Intelligent tool matcher using semantic embeddings for better task-tool matching.
    
    This class provides semantic similarity matching between tasks and tools,
    using the tool names and descriptions to find the most relevant tools.
    """
    
    def __init__(
        self, 
        max_tools: int = 3, 
        min_similarity: float = 0.1,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the tool matcher.
        
        Args:
            max_tools: Maximum number of tools to return for each match
            min_similarity: Minimum similarity threshold (0.0 to 1.0)
            model_name: Name of the sentence transformer model to use
        """
        self.max_tools = max_tools
        self.min_similarity = min_similarity
        self.model_name = model_name
        self.tools = []
        self.tool_embeddings = []
        self.model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the sentence transformer model."""
        try:
            # Check network connectivity first
            print("🌐 Checking network connectivity...")
            import urllib.request
            try:
                urllib.request.urlopen('https://huggingface.co', timeout=5)
                print("✅ Network connection available")
            except Exception as e:
                print("❌ Network connection failed, falling back to TF-IDF matching")
                print(f"   Error: {e}")
                self.model = None
                self._initialize_tfidf()
                return
            
            print(f"📥 Downloading sentence transformer model: {self.model_name}")
            print("   This may take a few minutes on first run...")
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Initialized sentence transformer model: {self.model_name}")
        except ImportError:
            print("⚠️  sentence-transformers not available, falling back to TF-IDF matching")
            self.model = None
            self._initialize_tfidf()
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            print("   Falling back to TF-IDF matching...")
            self.model = None
            self._initialize_tfidf()
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer as fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            print("✅ Initialized TF-IDF vectorizer as fallback")
        except ImportError:
            print("⚠️  scikit-learn not available, using simple keyword matching")
            self.tfidf_vectorizer = None
        except Exception as e:
            print(f"❌ Failed to initialize TF-IDF: {e}")
            self.tfidf_vectorizer = None
    
    def fit(self, tools: List[Dict[str, Any]]):
        """
        Fit the matcher with a list of tools.
        
        Args:
            tools: List of tool dictionaries with 'function' containing 'name' and 'description'
        """
        print(f"🚀 Starting tool matcher training with {len(tools)} tools...")
        self.tools = tools
        
        if self.model is not None:
            # Use sentence transformers
            print("🧠 Using sentence transformer model for semantic matching")
            self._fit_sentence_transformers()
        elif self.tfidf_vectorizer is not None:
            # Use TF-IDF
            print("📊 Using TF-IDF vectorizer for text matching")
            self._fit_tfidf()
        else:
            print("⚠️  No embedding model available, using simple text matching")
        
        print("🎯 Tool matcher training completed!")
    
    def _fit_sentence_transformers(self):
        """Fit using sentence transformers."""
        print("🔄 Training sentence transformer model (this may take a moment)...")
        tool_texts = []
        for tool in self.tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")
            tool_text = f"{name} {description}".strip()
            tool_texts.append(tool_text)
        
        if tool_texts:
            try:
                print(f"📝 Processing {len(self.tools)} tools...")
                self.tool_embeddings = self.model.encode(tool_texts, convert_to_tensor=True)
                print(f"✅ Generated sentence transformer embeddings for {len(self.tools)} tools")
            except Exception as e:
                print(f"❌ Failed to generate embeddings: {e}")
                self.tool_embeddings = []
    
    def _fit_tfidf(self):
        """Fit using TF-IDF vectorizer."""
        print("🔄 Training TF-IDF vectorizer...")
        tool_texts = []
        for tool in self.tools:
            function = tool.get("function", {})
            name = function.get("name", "")
            description = function.get("description", "")
            tool_text = f"{name} {description}".strip()
            tool_texts.append(tool_text)
        
        if tool_texts:
            try:
                print(f"📝 Processing {len(self.tools)} tools...")
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(tool_texts)
                print(f"✅ Generated TF-IDF vectors for {len(self.tools)} tools")
            except Exception as e:
                print(f"❌ Failed to generate TF-IDF vectors: {e}")
                self.tfidf_matrix = None
    
    def match_tools(self, task: str) -> List[Tuple[str, float]]:
        """
        Match a task with the most relevant tools.
        
        Args:
            task: The task description to match against tools
            
        Returns:
            List of tuples (tool_name, similarity_score) sorted by similarity
        """
        if not self.tools:
            return []
        
        if self.model is not None and len(self.tool_embeddings) > 0:
            # Use sentence transformers
            return self._match_sentence_transformers(task)
        elif self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
            # Use TF-IDF
            return self._match_tfidf(task)
        else:
            # Fallback to simple text matching
            return self._simple_match_tools(task)
    
    def _match_sentence_transformers(self, task: str) -> List[Tuple[str, float]]:
        """Match using sentence transformers."""
        try:
            # Generate embedding for the task
            task_embedding = self.model.encode([task], convert_to_tensor=True)
            
            # Calculate cosine similarities
            similarities = self._cosine_similarity(task_embedding, self.tool_embeddings)
            
            # Create list of (tool_name, similarity) tuples
            tool_scores = []
            for i, (tool, similarity) in enumerate(zip(self.tools, similarities)):
                if similarity >= self.min_similarity:
                    function = tool.get("function", {})
                    tool_name = function.get("name", f"tool_{i}")
                    tool_scores.append((tool_name, float(similarity)))
            
            # Sort by similarity score (descending) and limit results
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            return tool_scores[:self.max_tools]
            
        except Exception as e:
            print(f"❌ Error in sentence transformer matching: {e}")
            return self._simple_match_tools(task)
    
    def _match_tfidf(self, task: str) -> List[Tuple[str, float]]:
        """Match using TF-IDF vectors."""
        try:
            # Transform task to TF-IDF vector
            task_vector = self.tfidf_vectorizer.transform([task])
            
            # Calculate cosine similarities
            similarities = self._cosine_similarity_tfidf(task_vector, self.tfidf_matrix)
            
            # Create list of (tool_name, similarity) tuples
            tool_scores = []
            for i, (tool, similarity) in enumerate(zip(self.tools, similarities)):
                if similarity >= self.min_similarity:
                    function = tool.get("function", {})
                    tool_name = function.get("name", f"tool_{i}")
                    tool_scores.append((tool_name, float(similarity)))
            
            # Sort by similarity score (descending) and limit results
            tool_scores.sort(key=lambda x: x[1], reverse=True)
            return tool_scores[:self.max_tools]
            
        except Exception as e:
            print(f"❌ Error in TF-IDF matching: {e}")
            return self._simple_match_tools(task)
    
    def _simple_match_tools(self, task: str) -> List[Tuple[str, float]]:
        """
        Simple text-based matching as fallback when embeddings are not available.
        
        Args:
            task: The task description
            
        Returns:
            List of tuples (tool_name, similarity_score)
        """
        task_lower = task.lower()
        tool_scores = []
        
        for i, tool in enumerate(self.tools):
            function = tool.get("function", {})
            name = function.get("name", "").lower()
            description = function.get("description", "").lower()
            
            # Simple keyword matching
            score = 0.0
            if name in task_lower:
                score += 0.5
            if description and any(word in task_lower for word in description.split()):
                score += 0.3
            
            if score > 0:
                tool_scores.append((function.get("name", f"tool_{i}"), score))
        
        # Sort by score and limit results
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        return tool_scores[:self.max_tools]
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two tensors."""
        try:
            import torch
            if torch.is_tensor(a) and torch.is_tensor(b):
                # Normalize vectors
                a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
                b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
                # Calculate cosine similarity
                similarity = torch.mm(a_norm, b_norm.t())
                return similarity.squeeze().cpu().numpy()
        except ImportError:
            pass
        
        # Fallback to numpy if torch is not available
        try:
            a_np = a.cpu().numpy() if hasattr(a, 'cpu') else np.array(a)
            b_np = b.cpu().numpy() if hasattr(b, 'cpu') else np.array(b)
            
            # Normalize vectors
            a_norm = a_np / np.linalg.norm(a_np, axis=1, keepdims=True)
            b_norm = b_np / np.linalg.norm(b_np, axis=1, keepdims=True)
            
            # Calculate cosine similarity
            similarity = np.dot(a_norm, b_norm.T)
            return similarity.squeeze()
        except Exception as e:
            print(f"❌ Error calculating cosine similarity: {e}")
            return np.zeros(len(self.tools))
    
    def _cosine_similarity_tfidf(self, task_vector, tool_matrix):
        """Calculate cosine similarity between TF-IDF vectors."""
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(task_vector, tool_matrix)
            return similarities.squeeze()
        except ImportError:
            # Fallback to numpy-based calculation
            try:
                task_array = task_vector.toarray() if hasattr(task_vector, 'toarray') else np.array(task_vector)
                tool_array = tool_matrix.toarray() if hasattr(tool_matrix, 'toarray') else np.array(tool_matrix)
                
                # Normalize vectors
                task_norm = task_array / np.linalg.norm(task_array, axis=1, keepdims=True)
                tool_norm = tool_array / np.linalg.norm(tool_array, axis=1, keepdims=True)
                
                # Calculate cosine similarity
                similarities = np.dot(task_norm, tool_norm.T)
                return similarities.squeeze()
            except Exception as e:
                print(f"❌ Error calculating TF-IDF cosine similarity: {e}")
                return np.zeros(len(self.tools))
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Get semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if self.model is None:
            return 0.0
        
        try:
            embeddings = self.model.encode([text1, text2], convert_to_tensor=True)
            similarity = self._cosine_similarity(embeddings[0:1], embeddings[1:2])
            return float(similarity[0]) if isinstance(similarity, np.ndarray) else float(similarity)
        except Exception as e:
            print(f"❌ Error calculating semantic similarity: {e}")
            return 0.0
    
    def get_tool_by_name(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by its name.
        
        Args:
            tool_name: Name of the tool to find
            
        Returns:
            Tool dictionary if found, None otherwise
        """
        for tool in self.tools:
            function = tool.get("function", {})
            if function.get("name") == tool_name:
                return tool
        return None
    
    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools."""
        return self.tools.copy()
    
    def update_tools(self, tools: List[Dict[str, Any]]):
        """Update the tools list and regenerate embeddings."""
        self.fit(tools)
