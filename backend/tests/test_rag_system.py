"""Integration tests for RAGSystem.query() method handling content queries."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from rag_system import RAGSystem
from vector_store import SearchResults
from models import Course, Lesson, CourseChunk
from .test_utils import create_mock_deepseek_response, create_mock_tool_call


class TestRAGSystem:
    """Integration test cases for RAGSystem functionality"""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration"""
        config = Mock()
        config.CHUNK_SIZE = 800
        config.CHUNK_OVERLAP = 100
        config.CHROMA_PATH = "./test_chroma"
        config.EMBEDDING_MODEL = "test-embedding-model"
        config.MAX_RESULTS = 5
        config.DEEPSEEK_API_KEY = "test_api_key"
        config.DEEPSEEK_MODEL = "deepseek-chat"
        config.MAX_HISTORY = 2
        return config
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for RAGSystem"""
        mocks = {
            'document_processor': Mock(),
            'vector_store': Mock(),
            'ai_generator': Mock(),
            'session_manager': Mock()
        }
        return mocks
    
    @pytest.fixture
    def rag_system(self, mock_config, mock_components):
        """Create RAGSystem with mocked components"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:
            
            # Configure mocks to return our prepared mock objects
            mock_doc_proc.return_value = mock_components['document_processor']
            mock_vector_store.return_value = mock_components['vector_store']
            mock_ai_gen.return_value = mock_components['ai_generator']
            mock_session_mgr.return_value = mock_components['session_manager']
            
            rag = RAGSystem(mock_config)
            return rag, mock_components
    
    @pytest.mark.integration
    def test_query_successful_content_search(self, rag_system):
        """Test successful content query with tool-based search"""
        rag, mocks = rag_system
        
        # Mock AI generator response with tool execution
        mocks['ai_generator'].generate_response.return_value = (
            "Based on the search results, machine learning is a field of AI that uses algorithms to learn from data."
        )
        
        # Mock tool manager sources
        rag.tool_manager.get_last_sources = Mock(return_value=["ML Course - Lesson 1|https://example.com/ml/1"])
        rag.tool_manager.reset_sources = Mock()
        
        # Mock session manager
        mocks['session_manager'].get_conversation_history.return_value = None
        
        response, sources = rag.query("What is machine learning?")
        
        # Verify response
        assert "machine learning" in response.lower()
        assert "algorithms" in response.lower()
        
        # Verify sources
        assert len(sources) == 1
        assert "ML Course - Lesson 1" in sources[0]
        
        # Verify AI generator was called with correct parameters
        mocks['ai_generator'].generate_response.assert_called_once()
        call_args = mocks['ai_generator'].generate_response.call_args
        assert "machine learning" in call_args[1]['query'].lower()
        assert call_args[1]['tools'] is not None
        assert call_args[1]['tool_manager'] is not None
    
    @pytest.mark.integration
    def test_query_with_session_management(self, rag_system):
        """Test query with session ID and conversation history"""
        rag, mocks = rag_system
        
        session_id = "test_session_123"
        conversation_history = "User: Hello\nAssistant: Hi there!"
        
        # Mock session manager
        mocks['session_manager'].get_conversation_history.return_value = conversation_history
        
        # Mock AI response
        mocks['ai_generator'].generate_response.return_value = "Following up on our conversation..."
        
        # Mock tool manager
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        response, sources = rag.query("Tell me more about that topic", session_id=session_id)
        
        # Verify session management
        mocks['session_manager'].get_conversation_history.assert_called_once_with(session_id)
        mocks['session_manager'].add_exchange.assert_called_once_with(
            session_id,
            "Tell me more about that topic", 
            "Following up on our conversation..."
        )
        
        # Verify AI generator received history
        call_args = mocks['ai_generator'].generate_response.call_args[1]
        assert call_args['conversation_history'] == conversation_history
    
    @pytest.mark.integration
    def test_query_without_session(self, rag_system):
        """Test query without session ID"""
        rag, mocks = rag_system
        
        # Mock AI response
        mocks['ai_generator'].generate_response.return_value = "This is a response about deep learning."
        
        # Mock tool manager
        rag.tool_manager.get_last_sources = Mock(return_value=["Deep Learning Course"])
        rag.tool_manager.reset_sources = Mock()
        
        response, sources = rag.query("Explain deep learning")
        
        # Verify no session management calls
        mocks['session_manager'].get_conversation_history.assert_not_called()
        mocks['session_manager'].add_exchange.assert_not_called()
        
        # Verify AI generator called without history
        call_args = mocks['ai_generator'].generate_response.call_args[1]
        assert call_args['conversation_history'] is None
    
    @pytest.mark.integration
    def test_query_with_tool_execution_flow(self, rag_system):
        """Test complete query flow with tool execution"""
        rag, mocks = rag_system
        
        # Setup mock search tool in the tool manager
        mock_search_tool = Mock()
        mock_search_tool.last_sources = ["Neural Networks Course - Lesson 2|https://example.com/nn/2"]
        rag.search_tool = mock_search_tool
        
        # Mock AI generator to simulate tool calling
        mocks['ai_generator'].generate_response.return_value = (
            "Neural networks are computational models inspired by biological neural networks."
        )
        
        # Mock tool manager behavior
        rag.tool_manager.get_last_sources = Mock(return_value=["Neural Networks Course - Lesson 2|https://example.com/nn/2"])
        rag.tool_manager.reset_sources = Mock()
        
        response, sources = rag.query("How do neural networks work?")
        
        # Verify the complete flow
        assert "neural networks" in response.lower()
        assert len(sources) == 1
        assert "Neural Networks Course" in sources[0]
        
        # Verify tool definitions were passed to AI generator
        call_args = mocks['ai_generator'].generate_response.call_args[1]
        assert 'tools' in call_args
        assert call_args['tools'] is not None
        assert call_args['tool_manager'] is not None
        
        # Verify sources were retrieved and reset
        rag.tool_manager.get_last_sources.assert_called_once()
        rag.tool_manager.reset_sources.assert_called_once()
    
    @pytest.mark.integration
    def test_query_tool_execution_error(self, rag_system):
        """Test query when tool execution encounters an error"""
        rag, mocks = rag_system
        
        # Mock AI generator to return error-based response
        mocks['ai_generator'].generate_response.return_value = (
            "I apologize, but I encountered an error while searching for that information."
        )
        
        # Mock tool manager with empty sources (no successful search)
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        response, sources = rag.query("Find information about quantum computing")
        
        # Verify error handling
        assert "error" in response.lower() or "apologize" in response.lower()
        assert len(sources) == 0
        
        # Verify sources were still reset
        rag.tool_manager.reset_sources.assert_called_once()
    
    @pytest.mark.integration
    def test_add_course_document_integration(self, rag_system):
        """Test adding a course document through the system"""
        rag, mocks = rag_system
        
        # Mock document processor
        sample_course = Course(
            title="Test ML Course",
            course_link="https://test.com/ml",
            instructor="Dr. Test",
            lessons=[Lesson(lesson_number=0, title="Intro")]
        )
        
        sample_chunks = [
            CourseChunk(
                content="Introduction to machine learning concepts",
                course_title="Test ML Course",
                lesson_number=0,
                chunk_index=0
            )
        ]
        
        mocks['document_processor'].process_course_document.return_value = (sample_course, sample_chunks)
        
        course, chunk_count = rag.add_course_document("/test/path/course.txt")
        
        # Verify document processing
        mocks['document_processor'].process_course_document.assert_called_once_with("/test/path/course.txt")
        
        # Verify vector store operations
        mocks['vector_store'].add_course_metadata.assert_called_once_with(sample_course)
        mocks['vector_store'].add_course_content.assert_called_once_with(sample_chunks)
        
        # Verify return values
        assert course == sample_course
        assert chunk_count == 1
    
    @pytest.mark.integration
    def test_add_course_folder_integration(self, rag_system):
        """Test adding multiple course documents from folder"""
        rag, mocks = rag_system
        
        # Create temporary folder with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = ["course1.txt", "course2.txt", "course3.pdf"]
            for filename in test_files:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write("Test course content")
            
            # Mock existing course titles
            mocks['vector_store'].get_existing_course_titles.return_value = []
            
            # Mock document processor for each file
            def mock_process_document(file_path):
                basename = os.path.basename(file_path)
                course_title = f"Course from {basename}"
                course = Course(title=course_title, lessons=[])
                chunks = [CourseChunk(
                    content=f"Content from {basename}",
                    course_title=course_title,
                    chunk_index=0
                )]
                return course, chunks
            
            mocks['document_processor'].process_course_document.side_effect = mock_process_document
            
            total_courses, total_chunks = rag.add_course_folder(temp_dir, clear_existing=False)
            
            # Verify processing of valid files (txt and pdf)
            assert total_courses == 2  # course1.txt and course3.pdf
            assert total_chunks == 2
            
            # Verify vector store operations
            assert mocks['vector_store'].add_course_metadata.call_count == 2
            assert mocks['vector_store'].add_course_content.call_count == 2
    
    @pytest.mark.integration
    def test_get_course_analytics_integration(self, rag_system):
        """Test getting course analytics"""
        rag, mocks = rag_system
        
        # Mock vector store analytics
        mocks['vector_store'].get_course_count.return_value = 5
        mocks['vector_store'].get_existing_course_titles.return_value = [
            "Machine Learning Basics",
            "Deep Learning Advanced",
            "Neural Networks", 
            "Computer Vision",
            "Natural Language Processing"
        ]
        
        analytics = rag.get_course_analytics()
        
        # Verify analytics data
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Machine Learning Basics" in analytics["course_titles"]
        assert "Neural Networks" in analytics["course_titles"]
    
    @pytest.mark.integration
    def test_query_with_complex_session_flow(self, rag_system):
        """Test complex query flow with multiple interactions in session"""
        rag, mocks = rag_system
        
        session_id = "complex_session_456"
        
        # First query
        mocks['session_manager'].get_conversation_history.return_value = None
        mocks['ai_generator'].generate_response.return_value = "Machine learning is a subset of AI."
        rag.tool_manager.get_last_sources = Mock(return_value=["ML Intro Course"])
        rag.tool_manager.reset_sources = Mock()
        
        response1, sources1 = rag.query("What is machine learning?", session_id=session_id)
        
        # Second query with history
        conversation_history = "User: What is machine learning?\nAssistant: Machine learning is a subset of AI."
        mocks['session_manager'].get_conversation_history.return_value = conversation_history
        mocks['ai_generator'].generate_response.return_value = "Common algorithms include linear regression, decision trees, and neural networks."
        rag.tool_manager.get_last_sources = Mock(return_value=["ML Algorithms Course - Lesson 3"])
        
        response2, sources2 = rag.query("What are common ML algorithms?", session_id=session_id)
        
        # Verify both interactions
        assert "machine learning" in response1.lower()
        assert "algorithms" in response2.lower()
        
        # Verify session management for both calls
        assert mocks['session_manager'].add_exchange.call_count == 2
        
        # Verify second call used conversation history
        second_call_args = mocks['ai_generator'].generate_response.call_args_list[1][1]
        assert second_call_args['conversation_history'] == conversation_history
    
    @pytest.mark.integration
    def test_query_prompt_formatting(self, rag_system):
        """Test that query prompt is properly formatted"""
        rag, mocks = rag_system
        
        mocks['ai_generator'].generate_response.return_value = "Test response"
        rag.tool_manager.get_last_sources = Mock(return_value=[])
        rag.tool_manager.reset_sources = Mock()
        
        user_query = "Explain supervised learning"
        rag.query(user_query)
        
        # Verify the prompt formatting
        call_args = mocks['ai_generator'].generate_response.call_args[1]
        expected_prompt = f"Answer this question about course materials: {user_query}"
        assert call_args['query'] == expected_prompt
    
    @pytest.mark.integration  
    def test_system_component_initialization(self, mock_config):
        """Test that all system components are properly initialized"""
        with patch('rag_system.DocumentProcessor') as mock_doc_proc, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_gen, \
             patch('rag_system.SessionManager') as mock_session_mgr:
            
            rag = RAGSystem(mock_config)
            
            # Verify all components were initialized with correct parameters
            mock_doc_proc.assert_called_once_with(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
            mock_vector_store.assert_called_once_with(mock_config.CHROMA_PATH, mock_config.EMBEDDING_MODEL, mock_config.MAX_RESULTS)
            mock_ai_gen.assert_called_once_with(mock_config.DEEPSEEK_API_KEY, mock_config.DEEPSEEK_MODEL)
            mock_session_mgr.assert_called_once_with(mock_config.MAX_HISTORY)
            
            # Verify tools were registered
            assert hasattr(rag, 'tool_manager')
            assert hasattr(rag, 'search_tool')
            assert hasattr(rag, 'outline_tool')