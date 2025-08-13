# -*- coding: utf-8 -*-

import numpy as np
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class ToolMatcher:
    """Tool matcher for analyzing task-tool relevance"""
    
    def __init__(self, max_tools: int = 5):
        """
        Initialize tool matcher
        
        Args:
            max_tools: Maximum number of tools to match
        """
        self.max_tools = max_tools
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            min_df=1,  # Minimum document frequency
            max_df=1.0  # Maximum document frequency
        )
        self.tool_embeddings = None
        self.tool_names = []
        self.tool_descriptions = []
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text"""
        if not text:
            return ""
        # Remove special characters, keep letters, numbers and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _extract_tool_text(self, tool: Dict) -> str:
        """Extract text information from tool"""
        tool_name = tool.get("function", {}).get("name", "")
        tool_description = tool.get("function", {}).get("description", "")
        
        # Preprocess tool name and description
        name_text = self._preprocess_text(tool_name)
        desc_text = self._preprocess_text(tool_description)
        
        # Combine tool name and description, name has higher weight
        # Replace underscores with spaces in tool name to improve matching
        name_text = name_text.replace('_', ' ')
        combined_text = f"{name_text} {name_text} {desc_text}"  # Name repeated twice for higher weight
        return combined_text
    
    def fit(self, tools: List[Dict]) -> None:
        """
        Train tool matcher
        
        Args:
            tools: List of tools, each containing function.name and function.description
        """
        self.tool_names = []
        self.tool_descriptions = []
        tool_texts = []
        
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            tool_description = tool.get("function", {}).get("description", "")
            
            self.tool_names.append(tool_name)
            self.tool_descriptions.append(tool_description)
            
            # Extract tool text
            tool_text = self._extract_tool_text(tool)
            tool_texts.append(tool_text)
        
        # Train TF-IDF vectorizer
        if tool_texts:
            self.tool_embeddings = self.vectorizer.fit_transform(tool_texts)
    
    def match_tools(self, task: str) -> List[Tuple[str, float]]:
        """
        Match task with tools based on relevance
        
        Args:
            task: Task description
            
        Returns:
            List of matched tools, each element is (tool_name, similarity_score)
        """
        if self.tool_embeddings is None or not self.tool_names:
            return []
        
        # Preprocess task text
        task_text = self._preprocess_text(task)
        
        # Vectorize task
        task_vector = self.vectorizer.transform([task_text])
        
        # Calculate similarity
        similarities = cosine_similarity(task_vector, self.tool_embeddings).flatten()
        
        # Create tool name and similarity score pairs
        tool_scores = list(zip(self.tool_names, similarities))
        
        # Filter out tools with zero similarity, unless all tools are zero
        non_zero_scores = [score for score in tool_scores if score[1] > 0]
        if non_zero_scores:
            tool_scores = non_zero_scores
        
        # Sort by similarity score in descending order
        tool_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top max_tools tools
        return tool_scores[:self.max_tools]
    
    def get_relevant_tools(self, task: str, tools: List[Dict]) -> List[Dict]:
        """
        Get subset of tools relevant to the given task
        
        Args:
            task: Task description
            tools: Complete list of available tools
            
        Returns:
            Subset of relevant tools
        """
        # Retrain matcher (if tool list has changed)
        self.fit(tools)
        
        # Get matched tool names and scores
        matched_tools = self.match_tools(task)
        
        # Filter tools based on matched tool names
        relevant_tools = []
        matched_names = [name for name, _ in matched_tools]
        
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name in matched_names:
                relevant_tools.append(tool)
        
        return relevant_tools
    
    def get_match_scores(self, task: str) -> Dict[str, float]:
        """
        Get match scores for all tools against the given task
        
        Args:
            task: Task description
            
        Returns:
            Mapping of tool names to similarity scores
        """
        matched_tools = self.match_tools(task)
        return dict(matched_tools)
