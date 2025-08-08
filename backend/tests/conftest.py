import pytest
import sys
import os
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

# Add the backend directory to the Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=0,
            title="Introduction",
            lesson_link="https://example.com/lesson0"
        ),
        Lesson(
            lesson_number=1,
            title="Getting Started",
            lesson_link="https://example.com/lesson1"
        ),
        Lesson(
            lesson_number=2,
            title="Advanced Topics",
            lesson_link="https://example.com/lesson2"
        )
    ]
    
    return Course(
        title="Test Course on Machine Learning",
        course_link="https://example.com/course",
        instructor="Dr. Test",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    chunks = [
        CourseChunk(
            content="Course Test Course on Machine Learning Lesson 0 content: This is an introduction to machine learning concepts.",
            course_title="Test Course on Machine Learning",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="This lesson covers the basics of supervised learning algorithms.",
            course_title="Test Course on Machine Learning",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced topics include neural networks and deep learning frameworks.",
            course_title="Test Course on Machine Learning",
            lesson_number=2,
            chunk_index=2
        )
    ]
    return chunks


@pytest.fixture
def sample_search_results():
    """Create sample search results for testing"""
    return SearchResults(
        documents=[
            "Course Test Course on Machine Learning Lesson 0 content: This is an introduction to machine learning concepts.",
            "This lesson covers the basics of supervised learning algorithms."
        ],
        metadata=[
            {
                'course_title': 'Test Course on Machine Learning',
                'lesson_number': 0,
                'chunk_index': 0
            },
            {
                'course_title': 'Test Course on Machine Learning', 
                'lesson_number': 1,
                'chunk_index': 1
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def error_search_results():
    """Create search results with error for testing"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=["Test content about machine learning"],
        metadata=[{'course_title': 'Test Course', 'lesson_number': 1}],
        distances=[0.1]
    )
    mock_store._resolve_course_name.return_value = "Test Course on Machine Learning"
    mock_store.get_lesson_link.return_value = "https://example.com/lesson1"
    return mock_store


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client for testing"""
    mock_client = Mock()
    
    # Mock successful response without tool calls
    mock_response = Mock()
    mock_choice = Mock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.content = "This is a test response about machine learning concepts."
    mock_choice.message.tool_calls = None
    mock_response.choices = [mock_choice]
    
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_call_response():
    """Create a mock OpenAI response with tool calls"""
    mock_client = Mock()
    
    # Mock tool call response
    mock_tool_call = Mock()
    mock_tool_call.id = "call_123"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "search_course_content"
    mock_tool_call.function.arguments = '{"query": "machine learning", "course_name": "Test Course"}'
    
    mock_choice = Mock()
    mock_choice.finish_reason = "tool_calls"
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tool_call]
    
    mock_response = Mock()
    mock_response.choices = [mock_choice]
    
    # Mock final response after tool execution
    mock_final_choice = Mock()
    mock_final_choice.finish_reason = "stop"
    mock_final_choice.message.content = "Based on the search results, here's what I found about machine learning."
    
    mock_final_response = Mock()
    mock_final_response.choices = [mock_final_choice]
    
    # Set up client to return tool call response first, then final response
    mock_client.chat.completions.create.side_effect = [mock_response, mock_final_response]
    
    return mock_client, mock_tool_call


@pytest.fixture
def test_config():
    """Create a test configuration"""
    config = Config()
    config.DEEPSEEK_API_KEY = "test_api_key"
    config.DEEPSEEK_MODEL = "deepseek-chat"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_tool_definitions():
    """Sample tool definitions for testing"""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_course_content",
                "description": "Search course materials with smart course name matching",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "course_name": {"type": "string", "description": "Course title"},
                        "lesson_number": {"type": "integer", "description": "Lesson number"}
                    },
                    "required": ["query"]
                }
            }
        }
    ]