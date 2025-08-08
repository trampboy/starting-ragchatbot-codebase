"""Test utilities and helper functions for the RAG system tests."""

import tempfile
import os
import shutil
from typing import Dict, List, Any
from unittest.mock import Mock
import json


class MockChromaCollection:
    """Mock ChromaDB collection for testing"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        self.ids = []
        self.should_raise_error = False
        self.error_message = "Mock database error"
    
    def add(self, documents, metadatas, ids):
        """Mock add method"""
        if self.should_raise_error:
            raise Exception(self.error_message)
        
        self.documents.extend(documents)
        self.metadata.extend(metadatas)
        self.ids.extend(ids)
    
    def query(self, query_texts, n_results=5, where=None):
        """Mock query method"""
        if self.should_raise_error:
            raise Exception(self.error_message)
        
        # Simple mock logic - return stored documents if they exist
        if not self.documents:
            return {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
        
        # Apply basic filtering if where clause exists
        filtered_docs = []
        filtered_metadata = []
        filtered_distances = []
        
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadata)):
            include = True
            
            if where:
                if '$and' in where:
                    # Handle AND conditions
                    for condition in where['$and']:
                        for key, value in condition.items():
                            if key not in meta or meta[key] != value:
                                include = False
                                break
                        if not include:
                            break
                else:
                    # Handle single conditions
                    for key, value in where.items():
                        if key not in meta or meta[key] != value:
                            include = False
                            break
            
            if include:
                filtered_docs.append(doc)
                filtered_metadata.append(meta)
                filtered_distances.append(0.1 * i)  # Mock distance
                
                if len(filtered_docs) >= n_results:
                    break
        
        return {
            'documents': [filtered_docs],
            'metadatas': [filtered_metadata],
            'distances': [filtered_distances]
        }
    
    def get(self, ids=None):
        """Mock get method"""
        if self.should_raise_error:
            raise Exception(self.error_message)
        
        if ids:
            # Filter by specific IDs
            result_docs = []
            result_meta = []
            result_ids = []
            
            for id in ids:
                if id in self.ids:
                    idx = self.ids.index(id)
                    result_docs.append(self.documents[idx])
                    result_meta.append(self.metadata[idx])
                    result_ids.append(id)
            
            return {
                'documents': result_docs,
                'metadatas': result_meta,
                'ids': result_ids
            }
        
        # Return all documents
        return {
            'documents': self.documents,
            'metadatas': self.metadata,
            'ids': self.ids
        }


class MockChromaClient:
    """Mock ChromaDB client for testing"""
    
    def __init__(self):
        self.collections = {}
        self.should_raise_error = False
        self.error_message = "Mock client error"
    
    def get_or_create_collection(self, name, embedding_function=None):
        """Mock collection creation"""
        if self.should_raise_error:
            raise Exception(self.error_message)
        
        if name not in self.collections:
            self.collections[name] = MockChromaCollection()
        
        return self.collections[name]
    
    def delete_collection(self, name):
        """Mock collection deletion"""
        if name in self.collections:
            del self.collections[name]


def create_temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_")
    return temp_dir


def cleanup_temp_path(temp_path: str):
    """Clean up temporary directory"""
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)


def create_sample_documents():
    """Create sample documents for testing document processing"""
    doc1 = """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Dr. Smith

Lesson 0: Getting Started
Lesson Link: https://example.com/ml-course/lesson-0
Welcome to machine learning! This course will teach you the fundamentals.
Machine learning is a subset of artificial intelligence that focuses on algorithms.

Lesson 1: Supervised Learning  
Lesson Link: https://example.com/ml-course/lesson-1
Supervised learning involves training models on labeled data.
Common algorithms include linear regression and decision trees.
The goal is to make predictions on new, unseen data.

Lesson 2: Unsupervised Learning
Lesson Link: https://example.com/ml-course/lesson-2
Unsupervised learning finds patterns in data without labels.
Clustering and dimensionality reduction are key techniques.
K-means clustering is a popular unsupervised algorithm."""
    
    doc2 = """Course Title: Advanced Deep Learning
Course Link: https://example.com/dl-course
Course Instructor: Prof. Johnson

Lesson 0: Neural Network Basics
Neural networks are the foundation of deep learning.
They consist of layers of interconnected nodes called neurons.

Lesson 1: Convolutional Networks
CNNs are specialized for processing grid-like data such as images.
They use convolution operations to detect local features.

Lesson 2: Recurrent Networks
RNNs are designed for sequential data like text and time series.
They have memory capabilities through recurrent connections."""
    
    return {
        "ml_course.txt": doc1,
        "dl_course.txt": doc2
    }


def create_temp_document_files():
    """Create temporary document files for testing"""
    temp_dir = tempfile.mkdtemp(prefix="test_docs_")
    documents = create_sample_documents()
    
    file_paths = []
    for filename, content in documents.items():
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        file_paths.append(file_path)
    
    return temp_dir, file_paths


def assert_search_results_equal(result1, result2, check_distances=False):
    """Helper function to compare SearchResults objects"""
    assert result1.documents == result2.documents
    assert result1.metadata == result2.metadata
    assert result1.error == result2.error
    
    if check_distances:
        assert result1.distances == result2.distances


def create_mock_deepseek_response(content: str, tool_calls=None):
    """Create a mock DeepSeek API response"""
    mock_choice = Mock()
    mock_choice.message.content = content
    mock_choice.message.tool_calls = tool_calls
    mock_choice.finish_reason = "tool_calls" if tool_calls else "stop"
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    return mock_response


def create_mock_tool_call(call_id: str, function_name: str, arguments: Dict[str, Any]):
    """Create a mock tool call object"""
    mock_tool_call = Mock()
    mock_tool_call.id = call_id
    mock_tool_call.type = "function"
    mock_tool_call.function.name = function_name
    mock_tool_call.function.arguments = json.dumps(arguments)
    
    return mock_tool_call