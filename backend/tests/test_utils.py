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


def create_mock_sequential_responses(responses_data: List[Dict[str, Any]]):
    """
    Create a list of mock responses for sequential tool calling tests.
    
    Args:
        responses_data: List of dicts with keys:
            - content: Response content
            - tool_calls: Optional list of tool calls
            - finish_reason: Optional finish reason (defaults based on tool_calls)
    
    Returns:
        List of mock response objects
    """
    mock_responses = []
    
    for response_data in responses_data:
        content = response_data.get('content', '')
        tool_calls = response_data.get('tool_calls', [])
        finish_reason = response_data.get('finish_reason')
        
        # Auto-determine finish reason if not provided
        if finish_reason is None:
            finish_reason = "tool_calls" if tool_calls else "stop"
        
        mock_response = create_mock_deepseek_response(content, tool_calls)
        mock_response.choices[0].finish_reason = finish_reason
        mock_responses.append(mock_response)
    
    return mock_responses


def create_mock_sequential_tool_manager():
    """Create a mock tool manager for sequential testing"""
    mock_tool_manager = Mock()
    
    # Mock tools and sources tracking
    mock_tool_manager.tools = {}
    mock_tool_manager._sequential_sources = []
    
    # Mock method implementations
    def mock_execute_tool(tool_name, **kwargs):
        # Return different responses based on tool name for testing
        if tool_name == "get_course_outline":
            return "Course: ML Basics\nLesson 1: Introduction\nLesson 2: Algorithms"
        elif tool_name == "search_course_content":
            return "[ML Basics - Lesson 1]\nMachine learning is a subset of AI."
        else:
            return f"Mock result for {tool_name}"
    
    def mock_get_tool_definitions():
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_course_outline",
                    "description": "Get course outline",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "course_name": {"type": "string"}
                        },
                        "required": ["course_name"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "search_course_content",
                    "description": "Search course content",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "course_name": {"type": "string"}
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    
    def mock_get_last_sources():
        return mock_tool_manager._sequential_sources.copy()
    
    def mock_reset_sources():
        mock_tool_manager._sequential_sources = []
    
    def mock_set_sequential_sources(sources):
        mock_tool_manager._sequential_sources = sources.copy()
    
    # Set up mock methods
    mock_tool_manager.execute_tool = Mock(side_effect=mock_execute_tool)
    mock_tool_manager.get_tool_definitions = Mock(side_effect=mock_get_tool_definitions)
    mock_tool_manager.get_last_sources = Mock(side_effect=mock_get_last_sources)
    mock_tool_manager.reset_sources = Mock(side_effect=mock_reset_sources)
    mock_tool_manager.set_sequential_sources = Mock(side_effect=mock_set_sequential_sources)
    
    return mock_tool_manager


def assert_sequential_execution_calls(mock_client, expected_call_count: int):
    """Assert that the correct number of API calls were made during sequential execution"""
    actual_calls = mock_client.chat.completions.create.call_count
    assert actual_calls == expected_call_count, f"Expected {expected_call_count} API calls, got {actual_calls}"


def get_api_call_messages(mock_client, call_index: int = 0):
    """Get the messages from a specific API call for inspection"""
    calls = mock_client.chat.completions.create.call_args_list
    if call_index >= len(calls):
        raise IndexError(f"Call index {call_index} out of range, only {len(calls)} calls made")
    
    call_kwargs = calls[call_index][1]  # Get keyword arguments
    return call_kwargs.get('messages', [])